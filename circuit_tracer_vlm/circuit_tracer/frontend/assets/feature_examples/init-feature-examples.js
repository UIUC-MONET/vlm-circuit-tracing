window.initFeatureExamples = function({
  containerSel,
  showLogits = true,
  showExamples = true,
  hideStaleOutputs = false,
  mergeTokens = true,            // <— new flag (default keeps current behavior)
}){
  var visState = {
    isDev: 0,
    showLogits,
    showExamples,
    hideStaleOutputs,
    mergeTokens,                 // <—
    activeToken: null,
    feature: null,
    featureIndex: -1,
    chartRowTop: 16,
    chartRowHeight: 82,
  };

  var indexFileExistsCache = new Map()

  // set up dom and render fns
  var sel = containerSel.html('').append('div.feature-examples')
  if (visState.showLogits) sel.append('div.feature-example-logits')
  if (visState.showExamples) sel.append('div.feature-example-list')
  var renderAll = util.initRenderAll(['feature'])

  
  if (visState.showLogits) window.initFeatureExamplesLogits({renderAll, visState, sel})
  if (visState.showExamples) window.initFeatureExamplesList({renderAll, visState, sel})

  return {loadFeature, renderFeature}

  async function renderFeature(scan, featureIndex, layerIdx){
    if (visState.hideStaleOutputs) sel.classed('is-stale-output', 1)
    if (featureIndex == visState.featureIndex) return
    // load feature data and exit early if featureIndex has changed
    visState.featureIndex = featureIndex
    var feature = await loadFeature(scan, featureIndex, layerIdx)
    if (feature.featureIndex == visState.featureIndex){
      visState.feature = feature
      renderAll.feature()
      if (visState.hideStaleOutputs) sel.classed('is-stale-output', 0)
    }

    return feature
  }

  function hfUrl(scan, path) {
    const repoId = scan.split('@')[0]
    const revision = scan.split('@')[1] || 'main'
    return `https://huggingface.co/${repoId}/resolve/${revision}/features/${path}`
  }

  function indexFileExists(scan) {
    if (indexFileExistsCache.has(scan)) return indexFileExistsCache.get(scan)
    
    const promise = fetch(hfUrl(scan, 'index.json.gz'), { method: 'HEAD' })
      .then(response => response.ok)
      .catch(error => {
        if (error.status === 404) {
          return false
        } else {
          throw error
        }
      })
    
    indexFileExistsCache.set(scan, promise)
    return promise
  }

  async function loadFeatureFromBinary(scan, featureIndex) {
    const [layerIdx, featIdx] = util.cantorUnpair(featureIndex)
    const indexData = await util.getFile(hfUrl(scan, 'index.json.gz'))
    const offsets = indexData[layerIdx]['offsets']
    const binFilename = indexData[layerIdx]['filename']
    const startByte = offsets[featIdx]
    const endByte = offsets[featIdx + 1]
    
    if (!binFilename || !offsets) {
      throw new Error(`Feature ${featureIndex} not found in index`)
    }

    if (startByte === undefined || endByte === undefined || endByte <= startByte) {
      throw new Error(`Feature ${featureIndex} not found in index`)
    }

    return await util.getFile(hfUrl(scan, binFilename), true, 'bin', `bytes=${startByte}-${endByte - 1}`)
  }


  async function loadFeature(scan, featureIndex){
    if (scan.startsWith('./')) {
      var feature = await  util.getFile(`${scan}/${featureIndex}.json`)
    } else if (await indexFileExists(scan)){
      var feature = await loadFeatureFromBinary(scan, featureIndex)
    } else if (scan.includes('/')) {
      var feature = await util.getFile(hfUrl(scan, `${featureIndex}.json`))
    } else {
      var feature = await util.getFile(`./features/${scan}/${featureIndex}.json`)
    }

    if (feature.act_min === undefined) {
      feature.act_min = 0
      feature.act_max = 1.4
    }

    feature.colorScale = d3.scaleSequential(d3.interpolateOranges)
      .domain([feature.act_min, feature.act_max]).clamp(1)

    feature.featureIndex = featureIndex
    feature.scan = scan

    return feature
  }
}

window.init?.()
