# Chatbot
Chatbot using local PC, without using Google collaboratory.


(Google colaboratory を使う方法がネットで紹介されてますが、医療での応用を念頭に、自分のパソコンにインストール実験。本音は無料へのこだわり)
https://nlp.ist.i.kyoto-u.ac.jp/?Driving+domain+QA+datasets

https://nlp.ist.i.kyoto-u.ac.jp/nl-resource/Driving%20domain%20QA%20datasets/download_ddqa.cgi
から、
DDQA-1.0.tar.gz
をダウンロードします。このファイルの解凍は、7-Zip File Managerを使うか、LinuxまたはWindowsでmsysをインストール済みなら、
tar xzvf DDQA-1.0.tar.gz
で解凍。

筆者はPython 3.7では成功。3.9では失敗。
git clone https://github.com/huggingface/transformers
cd transformers
py -3.7 -m pip install --upgrade pip
py -3.7 -m pip install .
py -3.7 -m pip install -r requirements.txt
(多分、エラー)
py -3.7 -m pip install fugashi[unidic-lite] ipadic datasets

git lfs install

git clone https://huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking

DDQA-1.0.tar.gzから解凍して抽出された以下の2つのファイルをtransformersのフォルダに置く。
DDQA-1.0_RC-QA_train.json
DDQA-1.0_RC-QA_dev.json

(次の計算は長時間)
py -3.7  ./examples/legacy/question-answering/run_squad.py  --model_type=bert   --model_name_or_path=cl-tohoku/bert-base-japanese-whole-word-masking   --do_train     --do_eval    --train_file=DDQA-1.0_RC-QA_train.json  --predict_file=DDQA-1.0_RC-QA_dev.json  --per_gpu_train_batch_size 12  --learning_rate 3e-5  --num_train_epochs 10  --max_seq_length 384  --doc_stride 128  --overwrite_output_dir   --output_dir output/

エラーが出たら、エラーメッセージを頼りに、例えば、
torch
tqdm
等をインストールして再度実行。
(学習には、Core i7 11世代 CPUでも18日を要しました。)

cd ..
py -3.7 bert-cp37-2022-11-15.py
で実行。
2台目以降のパソコンには、上記の
py -3.7  ./examples/legacy/question...
を実行して暫く待ち、学習を開始してから、Ctrl+Cで動作を中止します。outputフォルダの中身をコピーしたら学習時間を削減出来ました。
