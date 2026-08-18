const IMAGE_SOFT = '<image_soft_token>';
const isImageSoft = t => t === IMAGE_SOFT;
const IMG_BASE = 'http://localhost:5001/image/';

const viewerImg = document.getElementById('image');
const viewer = new Viewer(viewerImg, {
  inline: false, // modal viewer
  transition: false,
  viewed() {
    viewer.zoomTo(1);
  },
});

const toSrc = (id) => {
  const path = String(id).replace('#', '/');
  return IMG_BASE + path;//'/preview';
};

window.initFeatureExamplesList = function({renderAll, visState, sel}){
  var sel = sel.select('.feature-example-list')
  renderAll.feature.fns.push(async () => {
    if (visState.feature.isDead) return sel.html(`Feature ${visState.feature.featureIndex} failed to load`)
    // // Put quantiles into cols to fill white space.
    // var cols = d3.range(Math.max(1, Math.floor(sel.node().offsetWidth/800))).map(d => [])
    // cols.forEach(col => col.y = 0)
    // visState.feature.examples_quantiles.forEach((d, i) => {
    //   var col = cols[d3.minIndex(cols, d => d.y)]
    //   col.push(d)
    //   col.y += d.examples.length + 2 // quantile header/whitespace is about 2× bigger than an example
    //   if (!i) col.y += 6
    // })
    // 
    var cols = [visState.feature.examples_quantiles]
    sel.html('')
      .appendMany('div.example-2-col', cols)
      .appendMany('div', d => d)
      .each(drawQuantile)    
  })

  function drawQuantile(quantile){
    var sel = d3.select(this)

    var quintileSel = sel.append('div.example-quantile')
      .append('span.quantile-title').text(quantile.quantile_name + ' ')

    sel.appendMany('div.ctx-container', quantile.examples).each(drawExample)
  }

  function maybeHexEscapedToBytes(token) { // -> number[]
    let ret = [];
    while (token.length) {
      if (/^\\x[0-9a-f]{2}/.exec(token)) {
        ret.push(parseInt(token.slice(2, 4), 16));
        token = token.slice(4);
      } else {
        ret.push(...new TextEncoder().encode(token[0]));
        token = token.slice(1);
      }
    }
    return ret;
  }
  function mergeHexEscapedMax(tokens, acts) {
    let ret = [];
    let i = 0;
    while (i < tokens.length) {
      let pushedMerge = false;

      // never start a hex-merge on an image soft token
      if (!isImageSoft(tokens[i]) && /^\x[0-9a-f]{2}/.exec(tokens[i])) {
        let maxAct = acts[i];
        let buf = maybeHexEscapedToBytes(tokens[i]);

        for (let j = i + 1; j < Math.min(i + 5, tokens.length); j++) {
          // stop merging if the next token is an image soft token
          if (isImageSoft(tokens[j])) break;

          maxAct = Math.max(maxAct, acts[j]);
          buf.push(...maybeHexEscapedToBytes(tokens[j]));
          try {
            let text = new TextDecoder("utf-8", { fatal: true })
              .decode(new Uint8Array(buf));
            ret.push({ token: text, act: maxAct, minIndex: i, maxIndex: j });
            i = j + 1;
            pushedMerge = true;
            break;
          } catch (e) {
            continue;
          }
        }
      }

      if (!pushedMerge) {
        ret.push({
          token: tokens[i],
          act: acts[i],
          minIndex: i,
          maxIndex: i,
        });
        i++;
      }
    }
    return ret;
  }


  function mergeConsecutiveSameActivations(tokens) {
    const merged = [];
    let currentGroup = null;

    for (let i = 0; i < tokens.length; i++) {
      const token = tokens[i];

      if (!currentGroup) {
        currentGroup = { ...token };
        continue;
      }

      const sameAct = currentGroup.act === token.act;
      const hasImageSoft =
        isImageSoft(currentGroup.token) || isImageSoft(token.token);

      if (sameAct && !hasImageSoft) {
        // allow merge only when neither side is <image_soft_token>
        currentGroup.token += token.token;
        currentGroup.maxIndex = token.maxIndex;
      } else {
        merged.push(currentGroup);
        currentGroup = { ...token };
      }
    }
    if (currentGroup) merged.push(currentGroup);
    return merged;
  }


  function drawExample(exp){
    const isImageSoft = t => t === '<image_soft_token>';

    const sel = d3.select(this).append('div')
      .st({opacity: exp.is_repeated_datapoint ? .4 : 1});
    const textSel = sel.append('div.text-wrapper');

    if (exp.image_references?.length) {
      const refs = sel.append('div').classed('image-references', true)
      refs.append('span').text('Images: ')
      refs.appendMany('a', exp.image_references)
        .text(d => d)
        .attr('href', d => /^https?:\/\//.test(d) ? d : null)
        .attr('target', d => /^https?:\/\//.test(d) ? '_blank' : null)
        .attr('rel', d => /^https?:\/\//.test(d) ? 'noopener noreferrer' : null)
    }

    // Build tokenData as you already do (with or without merging)
    let tokenData = mergeHexEscapedMax(exp.tokens, exp.tokens_acts_list);
    tokenData = mergeConsecutiveSameActivations(tokenData);

    // === NEW: attach the image-soft **IDs** per rendered span ===
    for (const d of tokenData) {
      const ids = [];
      // If tokens were merged, a span may cover multiple original indices.
      for (let k = d.minIndex; k <= d.maxIndex; k++) {
        if (isImageSoft(exp.tokens[k])) {
          if (exp.ids && exp.ids[k] != null) ids.push(exp.ids[k]); // use the JSON id
        }
      }
      d.imageSoftIds = ids;                   // array of IDs covered by this span
      d.imageSoftId  = ids.length ? ids[0] : null; // convenience: first one
    }

    const tokenSel = textSel.appendMany('span.token', tokenData)
      .text(d => d.token) // text for normal tokens; we’ll overwrite for image-soft spans
      .classed('is-image-soft', d => d.imageSoftIds.length > 0)
      .attr('data-image-soft-ids', d => d.imageSoftIds.join(','));

    // 2) Color spans with activation
    tokenSel.filter(d => d.act).st({ background: d => visState.feature.colorScale(d.act) });

    // 3) Replace image-soft spans’ text with inline <img>

    gallery = new Viewer(textSel.node());

    tokenSel
      .on('click', function (event, d) {
        if (!d.imageSoftIds || !d.imageSoftIds.length) return;

        // Choose which image to show; here we pick the first
        const nextSrc = toSrc(d.imageSoftIds[0]);

        // Wait for the new image to load, then update & show the viewer
        const onLoad = () => {
          viewer.update();  // let Viewer know the <img> changed
          viewer.show();    // open modal
          viewerImg.removeEventListener('load', onLoad);
        };

        // If src already set to this URL and cached, we'll still call update()+show() shortly
        if (viewerImg.src === nextSrc && viewerImg.complete) {
          viewer.update();
          viewer.show();
        } else {
          viewerImg.addEventListener('load', onLoad, { once: true });
          viewerImg.src = nextSrc;
        }
      });

    // Keep your existing image injection if you still want inline thumbnails in tokens:
    tokenSel
      .filter(d => d.imageSoftIds.length > 0)
      .html(d => d.imageSoftIds.map(id => {
        const path = String(id).replace('#', '/');
        const src  = IMG_BASE + path + "/preview";
        return `<img loading="lazy" class="image-soft-token" alt="image_soft_token" src="${src}" />`;
      }).join(''))
      .classed('has-image', true);

    const centerNode = tokenSel
      .filter(d => d.minIndex <= exp.train_token_ind && exp.train_token_ind <= d.maxIndex)
      .classed('train_token_ind', 1)
      .node();

    if (!centerNode) return;

    const resizeObserver = new ResizeObserver(() => {
      const selWidth = sel.node().offsetWidth;
      const nodeWidth = centerNode.offsetWidth;
      const nodeLeft = centerNode.offsetLeft;
      const leftOffset = (selWidth - nodeWidth)/2 - nodeLeft;
      textSel.translate([leftOffset, 0]);
    });
    resizeObserver.observe(sel.node());
    

    
  }

}

// window.initFeatureExample?.()
window.init?.()
