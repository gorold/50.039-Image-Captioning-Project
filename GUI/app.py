# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import time
import base64
import os
import glob

app = dash.Dash(__name__)

app.layout = html.Div(id = 'root', children=[
    html.H1(children = 'Image Captioning', style={'textAlign': 'center'}),
    html.Div(id = 'right-panel', 
             children = [dcc.Upload(id='upload-image', children = ['Drag and Drop or ', html.A('Select a File')], style={'width': '70%',
                                                                                                                         'height': '60px', 
                                                                                                                         'lineHeight': '60px',
                                                                                                                         'borderWidth': '1px',
                                                                                                                         'borderStyle': 'dashed',
                                                                                                                         'borderRadius': '5px',
                                                                                                                         'textAlign': 'center', 
                                                                                                                         'display': 'block',
                                                                                                                         'margin-left': 'auto',
                                                                                                                         'margin-right': 'auto'},
                                                                                                                         accept='image/*'),
                         html.Br(),
                         html.Div(id = 'left-panel', 
                            children = [dcc.Loading(id="loading-2", 
                                children=[html.Div([html.Div(id="output-image")])], type="circle")])]
                         )])

def save_file(name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join('GUI/assets/static', name), "wb") as fp:
        fp.write(base64.decodebytes(data))
    return os.path.join('GUI/assets/static', name)

@app.callback(Output("output-image", "children"), 
             [Input("upload-image", "filename"), 
             Input("upload-image", "contents")])
def update_output(uploaded_filenames, uploaded_file_contents):
    time.sleep(1)
    """Save uploaded files and regenerate the file list."""
    if uploaded_filenames is not None and uploaded_file_contents is not None:
        filepath = save_file(uploaded_filenames, uploaded_file_contents)
        test_base64 = base64.b64encode(open(filepath, 'rb').read()).decode('ascii')
        output = html.Div(children = [html.Img(id = 'output-image', 
                                               src = f'data:image/png;base64,{test_base64}', 
                                               style={'width': '40%',
                                                      'height': 'auto', 
                                                      'display': 'block',
                                                      'margin-left': 'auto',
                                                      'margin-right': 'auto'}), 
                                      html.H2(children = 'Text Here', style={'textAlign': 'center'} )])
        return output

if __name__ == '__main__':
    cwd = os.getcwd()
    app.run_server(debug=False)
    if len(os.listdir(f'{cwd}/GUI/assets/static')) != 0:
        for files in glob.glob(f'{cwd}/GUI/assets/static/*'):
            os.remove(files)