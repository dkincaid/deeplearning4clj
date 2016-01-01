(ns deeplearning4clj.word2vec
  (:import [com.esotericsoftware.kryo Kryo]
           [com.esotericsoftware.kryo.io Output Input]
           [java.io InputStream]
           [org.deeplearning4j.models.embeddings.loader WordVectorSerializer]
           [org.deeplearning4j.models.embeddings.wordvectors WordVectors]
           [org.deeplearning4j.models.word2vec Word2Vec]
           [org.deeplearning4j.models.word2vec Word2Vec$Builder]
           [org.deeplearning4j.models.word2vec.wordstore.inmemory InMemoryLookupCache]
           [org.deeplearning4j.plot BarnesHutTsne$Builder]
           [org.deeplearning4j.text.sentenceiterator SentenceIterator FileSentenceIterator]
           [org.deeplearning4j.text.sentenceiterator SentencePreProcessor]
           [org.deeplearning4j.text.tokenization.tokenizer Tokenizer TokenPreProcess]
           [org.deeplearning4j.text.tokenization.tokenizerfactory TokenizerFactory DefaultTokenizerFactory]
           [org.deeplearning4j.util SerializationUtils]
           [org.nd4j.linalg.api.ops.impl.accum.distances CosineSimilarity])
  (:require [clojure.string :as str]
            [taoensso.timbre :as timbre]
            [clojure.java.io :as io]))

(timbre/refer-timbre)

(def kryo (Kryo.))

(defn save-vectors
  "Save just the word vectors from the given model."
  [w2v filename]
  (WordVectorSerializer/writeWordVectors w2v filename))

(defn load-vectors
  "Load the word vectors from the given file."
  [filename]
  (WordVectorSerializer/loadTxtVectors (io/file filename)))

(defn save-model
  "Save a Word2Vec model to the given file."
  [w2v filename]
  (save-vectors w2v (str filename ".vectors"))
  (with-open [o (Output. (io/output-stream (str filename ".vocab")))]
    (.writeObject kryo o (.vocab w2v))))

(defn load-model
  "Load a Word2Vec model from the given name"
  [model-name]
  (let [w2v (load-vectors (str model-name ".vectors"))
        voc (with-open [i (Input. (io/input-stream (str model-name ".vocab")))]
              (.readObject kryo i InMemoryLookupCache))
        _ (.setVocab w2v voc)]
    w2v))

(defn aggregatePreProcessor
  "A sentence pre-processor that applies a list of pre-processors"
  [preprocessors]
  (reify SentencePreProcessor
    (preProcess [this sentence]
                (loop [pp preprocessors
                         processed sentence]
                    (if (empty? pp)
                      processed
                      (recur (rest pp) (.preProcess (first pp) processed)))))))

(defn tokenize
  "Tokenize on white space, only return tokens longer than 2 and replace tokens of the form #d+ with the string HASHNUM."
  [s]
  (->> (str/split s #"\s+")
       (filter #(> (count %) 2))))

(defn tokenizer
  "A tokenizer which utilizes a given function to perform the tokenization of the string."
  [tokenizer-fn s]
  (let [preprocessor (atom nil)
        tokens (tokenizer-fn s)
        next-token-idx (atom 0)]
   (reify Tokenizer
     (hasMoreTokens [this] (< @next-token-idx (dec (count tokens))))
     (countTokens [this] (count tokens))
     (nextToken [this] (do (swap! next-token-idx inc)
                             (if (nil? @preprocessor)
                               (nth tokens (dec @next-token-idx))
                               (.preProcess @preprocessor
                                            (nth tokens (dec @next-token-idx))))))
     (getTokens [this] (if (nil? @preprocessor)
                         tokens
                         (map #(.preProcess @preprocessor %) tokens)))
     (setTokenPreProcessor [this pp]
                           (reset! preprocessor pp)))))

(defn tokenizer-factory
  "Tokenizer factory that creates a tokenizer using the given function."
  [tokenizer-fn]
  (let [preprocessor (atom nil)]
   (reify TokenizerFactory
     (^Tokenizer create [this ^String s]
                 (tokenizer tokenizer-fn s))

     (^Tokenizer create [this ^InputStream s]
                 (tokenizer tokenizer-fn (str/join " " (line-seq s))))

     (setTokenPreProcessor [this pp]
                           (reset! preprocessor pp)))))

(defn word2vec
  "Create a Deeplearning4j Word2Vec object which uses the provided sentence iterator and tokenizer."
  [sent-iter tok {:keys [batch-size min-freq layer-size window-size]
                          :or {batch-size 1000
                               min-freq 5
                               layer-size 300
                               window-size 5}}]
  (-> (Word2Vec$Builder.)
      (.batchSize batch-size)
      (.sampling 1e-5)
      (.minWordFrequency min-freq)
      (.useAdaGrad false)
      (.layerSize layer-size)
      (.iterations 30)
      (.learningRate 0.025)
      (.minLearningRate 1e-2)
      (.negativeSample 10)
      (.windowSize window-size)
      (.iterate sent-iter)
      (.tokenizerFactory (tokenizer-factory tok))
      (.build)))

(defn fit-model
  "Training a Word2Vec model."
  [w2v]
  (info "Training the Word2Vec model.")
  (.fit w2v)
  w2v)

(defn fit-and-save-model
  [w2v output-file]
  (let [v (fit-model w2v)]
    (if (not (nil? v))
        (save-model v output-file))
    v))


