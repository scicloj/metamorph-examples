(ns sciloj.sklearn
  (:require [scicloj.metamorph.core :as morph]
            [tablecloth.pipeline :as tc]
            [tablecloth.api :as tc-api]

            [tech.v3.libs.smile.metamorph :as smile]
            [tech.v3.dataset.metamorph :as ds-mm]
            [tech.v3.libs.smile.nlp :as nlp]
            [tech.v3.ml.metamorph :as ml-mm]
            [scicloj.sklearn-clj.metamorph :as sklearn-mm]
            )


  )


(def pipe
  (morph/pipeline
   (ds-mm/set-inference-target :Score)
   (sklearn-mm/transform :feature-extraction.text :TfidfVectorizer {:max-features 10000})
   (sklearn-mm/transform :decomposition "PCA"  {:n-components 100})
   (sklearn-mm/estimate :linear-model :logistic-regression {})
   ))


(def fitted-pipe
  (pipe
   {:metamorph/mode :fit
    :metamorph/data
    (tc-api/dataset "test/data/reviews.csv.gz" {:key-fn keyword })}))

(->  fitted-pipe :metamorph/data :Score frequencies)

(def predicted-pipe
  (pipe
   (merge fitted-pipe
          {:metamorph/mode :transform
           :metamorph/data
           (tc-api/dataset "test/data/reviews.csv.gz" {:key-fn keyword })})))


(->  predicted-pipe :metamorph/data :Score frequencies)
(->
 (:metamorph/data predicted-pipe)
  :Score
 ;; (tc-api/column-names)
 ;; (tc-api/column-count)
 )
