#! /usr/bin/env bash
mkdir -p embeddings
urls="http://nlp.stanford.edu/data/glove.6B.zip http://nlp.stanford.edu/data/glove.twitter.27B.zip"
for url in $urls; do
    echo "Fetching $url"
    curl -L $url -O -s &
done
wait

mv *.zip embeddings/
unzip embeddings/glove.6B.zip -d embeddings/
unzip embeddings/glove.twitter.27B.zip -d embeddings/
