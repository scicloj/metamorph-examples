(ns scicloj.metamorph-examples.reviews
  (:require [notespace.api :as notespace]
            [notespace.kinds :as kind]
            [notespace.state :as state]))



^kind/hidden
(comment
  (notespace/render-static-html)
  (notespace/init-with-browser)
  (notespace/listen)

  )

["# NLP Machine Learning pipeline"]


(require '[scicloj.metamorph.core :as morph]
         '[tech.v3.libs.smile.metamorph :as smile]
         '[tech.v3.dataset.metamorph :as ds-mm]
         '[tech.v3.dataset.modelling :as ds-mod]
         '[tech.v3.dataset :as ds]
         '[tech.v3.libs.smile.nlp :as nlp]
         '[tech.v3.ml.metamorph :as ml-mm]
         )

["## One sequential pipeline"]

["In here we setup a pipeline for text classification.
The column to predict is the :Score, people gave in a product review.
The pipeline consists in a :
* count vectorize, which converts the text to a bag-of-words representation
* a bow-to-sparse transformer, which transform the bag-of-words (as a map) into
  the sparse format needed by the maxent model of Smile
  We choose the top 1000 words as the vocabulary size
* the maxent model, which can be trained on this data
"]

(def pipe
  (morph/pipeline
  (morph/lift ds/select-columns [:Text :Score])
   (smile/count-vectorize :Text :bow nlp/default-text->bow {})
   (smile/bow->sparse-array :bow :bow-sparse #(nlp/->vocabulary-top-n % 1000))
   (ds-mm/set-inference-target :Score)
   (morph/lift ds/select-columns [:bow-sparse :Score])
   (ml-mm/model {:p 1000
                 :model-type :maxent-multinomial
                 :sparse-column :bow-sparse})))

["First we split the data in test/train,"]

(def train-test-split
  (->
   (ds/->dataset "data/reviews.csv.gz" {:key-fn keyword })
   (ds-mod/train-test-split )))

["and run the pipeline fn in mode :fit with the train data.
This runs the pipeline including the training of the model."]

(def trained-ctx
  (pipe
   {:metamorph/mode :fit
    :metamorph/data (:train-ds train-test-split)
    }))




["For predicting on new data, we need to merge the predicted pipeline context (which contains the trained model),
and the new data and mode: transform"]

(def predicted-ctx
  (pipe
   (merge trained-ctx
          {:metamorph/mode :transform
           :metamorph/data (:test-ds train-test-split)
           })))



["Now  we have the prediction in the predicted contexts and can get the :Score column"]

(-> predicted-ctx :metamorph/data :Score
    seq frequencies)


["## Composed pipeline"]

["As each pipeline function returns an other function, we can simply
collect the pipeline steps in sequences and compose them."]

["In this pipeline we use as well an other transformer, namely Tf-Idf"]

["This defines the pre-processing pipeline."]

(def preprocess-pipe
  [(morph/lift ds/select-columns [:Text :Score])
   (smile/count-vectorize :Text :bow nlp/default-text->bow {})
   (smile/bow->tfidf :bow :tfidf)
   (smile/bow->sparse-array :tfidf :bow-sparse #(nlp/->vocabulary-top-n % 1000))])

["The we define the model pipeline."]

(def model-pipe
  [(ds-mm/set-inference-target :Score)
   (ml-mm/model {:p 1000
                 :model-type :maxent-multinomial
                 :sparse-column :bow-sparse})])

["Know we compose both to one full pipeline."]
(def full-pipe
  (apply morph/pipeline
         (concat preprocess-pipe
                 model-pipe)))

["Runing training .."]
(def trained-ctx
  (full-pipe
   {:metamorph/mode :fit
    :metamorph/data (:train-ds train-test-split)}))


["Running prediction .."]
(def predicted-ctx
  (full-pipe
   (merge
    trained-ctx
    {:metamorph/mode :transform
     :metamorph/data (:test-ds train-test-split)})))

["Result:"]

(-> predicted-ctx :metamorph/data :Score
    seq frequencies)





["## Alternative pipeline syntax: declarative"]

["The pipelines can be as well expressed as pure maps.
This is detailed here: https://scicloj.github.io/tablecloth/index.html#Pipeline"]

(defn select-columns [col-seq]
  (morph/lift ds/select-columns col-seq))


(def preprocess-pipe
  [[:select-columns [:Text :Score]]
   [:smile/count-vectorize :Text :bow nlp/default-text->bow {}]
   [:smile/bow->tfidf :bow :tfidf]
   [:smile/bow->sparse-array :tfidf :bow-sparse #(nlp/->vocabulary-top-n % 1000)]])

(def model-pipe
  [[:ds-mm/set-inference-target :Score]
   [:ml-mm/model {:p 1000
                  :model-type :maxent-multinomial
                  :sparse-column :bow-sparse}]] )

(def full-pipe
  (morph/->pipeline
   (concat preprocess-pipe
           model-pipe)))

(def trained-ctx
  (full-pipe
   {:metamorph/mode :fit
    :metamorph/data (:train-ds train-test-split)}))

(def predicted-ctx
  (full-pipe
   (merge
    predicted-ctx
    {:metamorph/mode :transform
     :metamorph/data (:test-ds train-test-split)})))


(-> predicted-ctx :metamorph/data :Score
    seq frequencies)
