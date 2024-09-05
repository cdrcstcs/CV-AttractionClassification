from flask import Flask, request, jsonify
from model import classify_new_image
app = Flask(__name__)
@app.route('/classify_image', methods=['POST'])
def process_data():
    data = request.json
    encoded_image = data.get('encoded_image')
    if encoded_image:
        try:
            decoded_image = base64.b64decode(encoded_image)
            image = Image.open(BytesIO(decoded_image))
            predicted_label = classify_new_image(image)
            return jsonify({'predicted_label': predicted_label})
        except Exception as e:
            return jsonify({'error': 'Failed to decode image', 'message': str(e)}), 400
    else:
        return jsonify({'error': 'No encoded image provided'}), 400
if __name__ == '__main__':
    app.run(debug=True)