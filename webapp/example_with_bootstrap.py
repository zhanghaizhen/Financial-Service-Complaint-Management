from flask import Flask, render_template
app = Flask(__name__)

# home page
@app.route('/')
def index():
    return render_template('jumbotron.html', title='Hello!')

@app.route('/more/')
def more():
    return render_template('starter_template.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
