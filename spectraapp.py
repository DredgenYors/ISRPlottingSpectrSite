from flask import Flask, render_template, send_file
import matplotlib.pyplot as plt
import io

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('spectrasite.html')

@app.route('/plot_page')
def plot_page():
    return render_template('plot_page.html')

@app.route('/plot')
def plot():
    # Generate the plot
    plt.figure()
    plt.plot([0, 1, 2, 3], [10, 20, 25, 30])
    plt.title('Spectra Plot')

    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)