(ns deeplearning4clj.word2vec-test
  (:require [deeplearning4clj.word2vec :as sut]
            [midje.sweet :refer :all]))

(facts "About tokenization functions"
  (fact "Tokenize regex"
    (sut/tokenize-regex #":" "this:that:the other") => ["this" "that" "the other"])
  (fact "Tokenize whitespace"
    (sut/tokenize-whitespace "this that the other") => ["this" "that" "the" "other"]))

(fact "Lower case sentence pre-processor"
  (.preProcess sut/lowerCaseSentence "This is one.") => "this is one.")

(fact "Lower case token pre-processor"
  (.preProcess sut/lowerCaseToken "This") => "this")

(declare my-fn)

(facts "About tokenizers"
  (prerequisite (my-fn ...string...) => ["This" "that" "the" "other"])
  (let [t (sut/tokenizer my-fn ...string...)]
    (.hasMoreTokens t) => truthy
    (.countTokens t) => 4
    (.nextToken t) => "This"
    (.getTokens t) => ["This" "that" "the" "other"]
    (let [_ (.setTokenPreProcessor t sut/lowerCaseToken)]
      (.getTokens t) => ["this" "that" "the" "other"])))
