from flask import Flask, request, render_template_string, send_file, redirect, url_for, flash
import os
import tempfile
from werkzeug.utils import secure_filename
import TOKENIZE, POSTAG, CHECK
import re

app = Flask(__name__)
app.secret_key = 'dev'

BASE_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>ELAN Auto Tokenizer for the Shughni Project</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      background: #f9f9f9;
      color: #333;
      margin: 0;
      padding: 0;
    }
    .container {
      max-width: 900px;
      margin: 40px auto;
      background: #fff;
      padding: 24px;
      border-radius: 8px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    }
    h1 {
      margin-top: 0;
      font-size: 24px;
    }
    p {
      line-height: 1.5;
      font-size: 14px;
    }
    form {
      margin-top: 20px;
    }
    label {
      display: block;
      margin-bottom: 8px;
      cursor: pointer;
    }
    input[type=file] {
      display: block;
      margin: 12px 0;
    }
    input[type=submit] {
      background: #007bff;
      color: #fff;
      border: none;
      padding: 10px 18px;
      border-radius: 4px;
      cursor: pointer;
      font-size: 14px;
    }
    input[type=submit]:hover {
      background: #0056b3;
    }
    .output-line {
      font-family: Consolas;
      white-space: pre;
      margin: 0 0 6px;
      font-size: 14px;
    }
  </style>
</head>
<body>
<div class="container">
  {{ content | safe }}
</div>
</body>
</html>
"""

INDEX_CONTENT = """
<h1>ELAN Auto Tokenizer for the Shughni Project</h1>
<p>Choose the action and upload the EAF file:</p>
<form method=post enctype=multipart/form-data action="/process">
  <label><input type=radio name=action value=tokenize checked> (1) Tokenize and suggest glosses</label>
  <label><input type=radio name=action value=postag> (2) Suggest POS tags</label>
  <label><input type=radio name=action value=check> (3) Final check</label>
  <input type=file name=file accept=".eaf" required>
  <input type=submit value=Upload>
</form>
"""



@app.route('/')
def index():
    return render_template_string(BASE_TEMPLATE, content=INDEX_CONTENT)


@app.route('/process', methods=['POST'])
def process():
    f = request.files.get('file')
    if not f:
        flash('No file')
        return redirect(url_for('index'))
    filename = secure_filename(f.filename)
    if not filename.lower().endswith('.eaf'):
        flash('Only .eaf files allowed')
        return redirect(url_for('index'))
    action = request.form.get('action')
    tmpdir = tempfile.mkdtemp()
    in_path = os.path.join(tmpdir, filename)
    f.save(in_path)

    if action == 'tokenize':
        out_path = TOKENIZE.gloss_text(in_path)
        return send_file(out_path, as_attachment=True, download_name=os.path.basename(out_path))
    elif action == 'postag':
        out_path = POSTAG.postag_text(in_path)
        return send_file(out_path, as_attachment=True, download_name=os.path.basename(out_path))
    elif action == 'check':
        output_lines = CHECK.check_text(in_path)
        lines_html = ''.join(f'<p class="output-line">{line}</p>' for line in output_lines)
        content = (
            '<h1>ELAN Auto Tokenizer for the Shughni Project</h1>'
            '<p>Check results:</p>'
            '<hr>'
            f'{lines_html}'
        )
        return render_template_string(BASE_TEMPLATE, content=content)


if __name__ == '__main__':
    app.run(debug=True)
