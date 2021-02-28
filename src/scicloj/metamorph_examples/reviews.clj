(ns scicloj.metamorph-examples.reviews
  (:require [notespace.api :as notespace]
            [notespace.kinds :as kind]
            [notespace.state :as state]))


^kind/hidden
(comment
  (notespace/render-static-html)
  )

(require '[scicloj.metamorph.core :as morph]
         '[tech.v3.libs.smile.metamorph :as smile]
         '[tech.v3.dataset.metamorph :as ds-mm]
         '[tech.v3.dataset.modelling :as ds-mod]
         '[tech.v3.dataset :as ds]
         '[tech.v3.libs.smile.nlp :as nlp]
         '[tech.v3.ml.metamorph :as ml-mm]
         )



(def pipe
  (morph/pipeline
   (ds-mm/select-columns [:Text :Score])
   (smile/count-vectorize :Text :bow nlp/default-text->bow {})
   (smile/bow->sparse-array :bow :sparse #(nlp/->vocabulary-top-n % 1000))
   (ds-mm/set-inference-target :Score)
   (ds-mm/select-columns [:sparse :Score])
   (ml-mm/model {:p 1000
                 :model-type :maxent-multinomial
                 :sparse-column :sparse})))


(def train-test-split
  (->
   (ds/->dataset "data/reviews.csv.gz" {:key-fn keyword })
   (ds-mod/train-test-split )
   ))


(def trained-ctx
  (pipe
   {:metamorph/mode :fit
    :metamorph/data (:train-ds train-test-split)
    }))

(def predicted-ctx
  (pipe
   (merge trained-ctx
           {:metamorph/mode :transform
           :metamorph/data (:test-ds train-test-split)
           })))


(-> predicted-ctx
    :metamorph/data
    :Score
    seq
    frequencies)




^kind/hidden
(comment
  (def preprocess-pipe
    [
     (tc/select-columns [:Text :Score])
     (smile/count-vectorize :Text :bow nlp/default-text->bow {})
     (smile/bow->tf-idf :bow :tfidf)
     (smile/bow->sparse-array :tfidf :sparse #(nlp/->vocabulary-top-n % 1000))
     ])

  (def model-pipe
    [
     (ds/set-inference-target :Score)
     (model-maxent {:model-type :maxent-multinomial
                    :sparse-column :sparse})
     ])

  (def full-pipe
    (apply morph/pipeline
           (concat preprocess-pipe
                   model-pipe)))

  (def trained-pipeline
    (full-pipe
     {:metamorph/mode :fit
      :metamorph/data
      (tc-api/dataset "test/data/reviews.csv.gz" {:key-fn keyword })}))


  (def predicted-pipeline
    (full-pipe
     (merge
      trained-pipeline
      {:metamorph/mode :transform
       :metamorph/data
       (tc-api/dataset "test/data/reviews.csv.gz" {:key-fn keyword })})))

  )





^kind/hidden
(comment


  (def preprocess-pipe
    [
     [:tc/select-columns [:Text :Score]]
     [:smile/count-vectorize :Text :bow nlp/default-text->bow {}]
     [:smile/bow->tf-idf :bow :tfidf]
     [:smile/bow->sparse-array :tfidf :sparse #(nlp/->vocabulary-top-n % 1000)]
     ])

  (def model-pipe
    [
     [:ds/set-inference-target :Score]
     [:model-maxent {:model-type :maxent-multinomial
                     :sparse-column :sparse}]])

  (def full-pipe
    (morph/->pipeline
     (concat preprocess-pipe
             model-pipe)))

  (def trained-pipeline
    (full-pipe
     {:metamorph/mode :fit
      :metamorph/data
      (tc-api/dataset "test/data/reviews.csv.gz" {:key-fn keyword })}))



  )
