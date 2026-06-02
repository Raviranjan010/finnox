from finbert.finbert import SentimentEngine
import argparse
import os


parser = argparse.ArgumentParser(description='Sentiment analyzer')

parser.add_argument('-a', action="store_true", default=False)

parser.add_argument('--text_path', type=str, help='Path to the text file.')
parser.add_argument('--output_dir', type=str, help='Where to write the results')
parser.add_argument('--model_path', type=str, help='Path to classifier model')

args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)


with open(args.text_path,'r') as f:
    text = f.read()

model = SentimentEngine(args.model_path)

output = "predictions.csv"
model.predict(text,write_to_csv=True,path=os.path.join(args.output_dir,output))