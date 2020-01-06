from flask import Flask
from flask_sse import sse
from flask import Response
from flask_cors import CORS
import time

app = Flask(__name__)
CORS(app)

@app.route("/")
def hello():
    return "Hello World!"
    
def get_message():
#'''this could be any function that blocks until data is ready'''
    time.sleep(1.0)
    s = time.ctime(time.time())
    print(s)
    return s

@app.route('/stream')
def stream():
    def eventStream():
        while True:
            # wait for source data to be available, then push it
            yield 'data: {}\n\n'.format(get_message())
    return Response(eventStream(), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run()
