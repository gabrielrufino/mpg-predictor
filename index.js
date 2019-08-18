const axios        = require('axios')
const readlineSync = require('readline-sync')
const tf           = require('@tensorflow/tfjs-node')

const getData = async () => {
  const { data: cars } = await axios.get('https://storage.googleapis.com/tfjs-tutorials/carsData.json')

  const cleaned = cars
    .map(car => ({
      mpg: car.Miles_per_Gallon,
      horsepower: car.Horsepower
    }))
    .filter(car => (car.mpg != null && car.horsepower != null))

  return cleaned
}

const convertToTensor = (data) => {
  return tf.tidy(() => {
    tf.util.shuffle(data)

    const inputs = data.map(d => d.horsepower)
    const labels = data.map(d => d.mpg)

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1])
    const labelTensor = tf.tensor2d(labels, [labels.length, 1])

    const inputMax = inputTensor.max()
    const inputMin = inputTensor.min()
    const labelMax = labelTensor.max()
    const labelMin = labelTensor.min()

    const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin))
    const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin))

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    }
  })
}

const createModel = () => {
  const model = tf.sequential()

  model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}))
  model.add(tf.layers.dense({units: 1, useBias: true}))

  return model
}

const trainModel = async (model, inputs, labels) => {
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse'],
  })
  
  const batchSize = 32;
  const epochs = 50;
  
  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true
  });
}

const main = async () => {
  const data = await getData()
  const tensorData = convertToTensor(data)
  const model = createModel()
  
  await trainModel(model, tensorData.inputs, tensorData.labels)

  return model
} 

main()
  .then(model => {})
