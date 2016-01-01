(defproject deeplearning4clj "0.1.0-SNAPSHOT"
  :description "A library for working with deeplearning4j from Clojure."
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.7.0"]
                 [org.deeplearning4j/deeplearning4j-core "0.4-rc3.7"]
                 [org.deeplearning4j/deeplearning4j-ui "0.4-rc3.7"]
                 [org.deeplearning4j/deeplearning4j-nlp "0.4-rc3.7"]
                 [org.nd4j/nd4j-api "0.4-rc3.7"]
                 [org.nd4j/nd4j-jblas "0.4-rc3.6"]
                 [com.taoensso/timbre "4.2.0"]
                 [com.esotericsoftware/kryo "3.0.3"]]
  :codox {:output-path "doc"}
  :profiles {:dev {:dependencies [[midje "1.8.3"]]
                   :plugins [[lein-midje "3.0.0"]]}})
