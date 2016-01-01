(ns deeplearning4clj.word2vec
  (:require [clojure.java.io :as io]
            [clojure.string :as str]
            [taoensso.timbre :as timbre])
  (:import [com.esotericsoftware.kryo.io Input Output]
           com.esotericsoftware.kryo.Kryo
           java.io.InputStream
           org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
           org.deeplearning4j.models.word2vec.Word2Vec$Builder
           org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache
           org.deeplearning4j.text.sentenceiterator.SentencePreProcessor
           [org.deeplearning4j.text.tokenization.tokenizer Tokenizer TokenPreProcess]
           org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory))

(timbre/refer-timbre)

(def kryo (Kryo.))

(defn save-vectors
  "Save just the word vectors from the given model to the file."
  [w2v filename]
  (WordVectorSerializer/writeWordVectors w2v filename))

(defn load-vectors
  "Load the word vectors from the given file."
  [filename]
  (WordVectorSerializer/loadTxtVectors (io/file filename)))

(defn save-model
  "Save a Word2Vec model to the given file. The model is saved in two files both prefixed by the given filename. The vectors
are saved in <filename>.vectors and the vocab is saved in <filename>.vocab. Uses Kryo to serialize the vocab object."
  [w2v filename]
  (save-vectors w2v (str filename ".vectors"))
  (with-open [o (Output. (io/output-stream (str filename ".vocab")))]
    (.writeObject kryo o (.vocab w2v))))

(defn load-model
  "Load a Word2Vec model from the given model name. The model name should be the file path to the location of the .vectors and .vocab
files that were written by save-model."
  [model-name]
  (let [w2v (load-vectors (str model-name ".vectors"))
        voc (with-open [i (Input. (io/input-stream (str model-name ".vocab")))]
              (.readObject kryo i InMemoryLookupCache))
        _ (.setVocab w2v voc)]
    w2v))


(def lowerCaseSentence
  "A sentence pre-processor that lower cases the sentence"
  (reify SentencePreProcessor
    (preProcess [this sentence]
                (str/lower-case sentence))))

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

(def lowerCaseToken
  "A token pre-processor that lower cases the token"
  (reify TokenPreProcess
    (preProcess [this token]
                (str/lower-case token))))

(defn tokenize-regex
  "Tokenize on the given regular expression"
  [regex s]
  (str/split s regex))

(def tokenize-whitespace
  "Tokenize on whitespace."
  (partial tokenize-regex #"\s+"))

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
  "Create a Deeplearning4j Word2Vec object which uses the provided sentence iterator and tokenizer. Keyword arguments can be used to set the batch size, minimum frequency, layer size and window size. All have good starting points as defaults."
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
  "Training a Word2Vec model. Returns the model."
  [w2v]
  (info "Training the Word2Vec model.")
  (.fit w2v)
  w2v)

(defn fit-and-save-model
  "Fit the model and save it to the given file. Returns the model."
  [w2v output-file]
  (let [v (fit-model w2v)]
    (if (not (nil? v))
        (save-model v output-file))
    v))

(defn top-n-words
  "Return the top n most frequent words from the VocabCache."
  [vocab n]
  (let [words (.words vocab)
        word-freqs (into {} (map #(hash-map % (.wordFrequency vocab %)) words))]
    (take n
          (into (sorted-map-by (fn [key1 key2]
                                 (compare [(get word-freqs key2) key2]
                                          [(get word-freqs key1) key1])))
                word-freqs))
    ))

(defn model-summary
  "Summary of a Word2Vec model."
  [model]
  (let [voc (.vocab model)]
    {;:window (.getWindow model)
     :numWords (.numWords voc)
     :totalWordOccurrences (.totalWordOccurrences voc)
     :totalNumberOfDocs (.totalNumberOfDocs voc)
     :topTenWords (top-n-words voc 10)
     }))

(defn add-vectors
  "Element-wise addition of two or more vectors."
  [& vs]
  (reduce #(.add %1 %2) vs))

(defn add-words
  "Adds the vectors of the words."
  [model & ws]
  (let [vs (map #(.getWordVectorMatrix model %) ws)]
    (apply add-vectors vs)))

(defn mult-vectors
  "Element-wise multiplication of two or more vectors."
  [& vs]
  (reduce #(.muli %1 %2) vs))

(defn mean-vectors
  "Mean of the given vectors"
  [& vs]
  (let [n (+ 1 (count vs))]
    (.divi (apply add-vectors vs) n)))

(defn mean-words
  "Mean of the word vectors for the given words"
  [model & ws]
  (let [vs (map #(.getWordVectorMatrix model %) ws)]
    (apply mean-vectors vs)))

(defn combine-vectors
  "Combine the given word vectors using the provided function."
  [f vs]
  (reduce f [] vs))

