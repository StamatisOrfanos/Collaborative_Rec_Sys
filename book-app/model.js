const tf = require('@tensorflow/tfjs-node')
const books = require("./data/web_book_data.json")



async function loadModel() {
    console.log('Loading Model...')
    const model = await tf.loadLayersModel("file:///home/stamatiosorfanos/Documents/Recommendation_System/book-app/model/model.json", false);
    console.log('Model Loaded Successfull')
    // console.log(model.summary())
    return model
}

const book_arr = tf.range(0, books.length)
const book_len = books.length


exports.recommend = async function recommend(userId) {
    let user = tf.fill([book_len], Number(userId))
    let book_in_js_array = book_arr.arraySync()
    let model = null;
    try {
        model = await loadModel()
    }catch(error) {
        console.log("beep boop")
        console.error(error)
    }
    console.log(`Recommending for User: ${userId}`)
    pred_tensor = await model.predict([book_arr, user]).reshape([50000]).catch()
    pred = pred_tensor.arraySync()
    let recommendations = []
    for (let i = 0; i < 6; i++) {
        max = pred_tensor.argMax().arraySync()
        recommendations.push(books[max]) //Push book with highest prediction probability
        pred.splice(max, 1)             //drop from array
        pred_tensor = tf.tensor(pred)  //create a new tensor
    }
    return recommendations
}