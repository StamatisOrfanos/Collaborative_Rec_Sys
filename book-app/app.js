const express = require("express");
const bodyParser = require("body-parser");
const expressHbs = require("express-handlebars")
const tf = require('@tensorflow/tfjs-node')
const books = require("./data/web_book_data.json")
const book_len = books.length

const book_arr = tf.range(0, books.length)



const app = express();
app.set("views", "./views");
app.set("view engine", "hbs");

//Body parser middleware
app.use(
    bodyParser.urlencoded({
        extended: false
    })
);
app.use(bodyParser.json());


app.engine('.hbs', expressHbs({
    defaultLayout: 'layouts',
    extname: '.hbs'
}));


app.get("/", (req, res) => {
    res.render("index", { books: books.slice(0, 10), pg_start: 0, pg_end: 10 })
});


app.get("/get-next", (req, res) => {
    let pg_start = Number(req.query.pg_end)
    let pg_end = Number(pg_start) + 10
    res.render("index", {
        books: books.slice(pg_start, pg_end),
        pg_start: pg_start,
        pg_end: pg_end
    })
});


app.get("/get-prev", (req, res) => {
    let pg_end = Number(req.query.pg_start)
    let pg_start = Number(pg_end) - 10

    if (pg_start <= 0) {
        res.render("index", { books: books.slice(0, 10), pg_start: 0, pg_end: 10 })

    } else {
        res.render("index", {
            books: books.slice(pg_start, pg_end),
            pg_start: pg_start,
            pg_end: pg_end
        })

    }
});

app.get("/recommend", (req, res) => {
    let userId = req.query.userId
    if (Number(userId) > 53424 || Number(userId) < 0) {
        res.send("User Id cannot be greater than 53.424 or less than 0!")
    } else {
        tf.loadLayersModel("file:///home/stamatiosorfanos/Documents/Recommendation_System/book-app/model/model.json", false).then(model => {
            let user = tf.fill([book_len], Number(userId))
            let book_in_js_array = book_arr.arraySync()
            console.log(`Recommending for User: ${userId}`)
            let pred_tensor = model.predict([book_arr, user])
            pred = pred_tensor.reshape([50000]).arraySync()
            let recommendations = []
            for (let i = 0; i < 6; i++) {
                max = pred_tensor.argMax().arraySync()
                recommendations.push(books[max]) //Push book with highest prediction probability
                pred.splice(max, 1)             //drop from array
                pred_tensor = tf.tensor(pred)  //create a new tensor
            }
            
            console.log(recommendations)
            res.render("index", {recommendations: recommendations, forUser: true})

        })
    }
})

module.exports = app;