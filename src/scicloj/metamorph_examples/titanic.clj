(ns scicloj.metamorph-examples.titanic
  (:require
   [notespace.api :as note]
   [notespace.kinds :as kind ]))

(comment
  (note/init-with-browser)
  (note/eval-this-notespace)
  (note/reread-this-notespace)
  (note/render-static-html)
  (note/init)
  )


(require '[scicloj.metamorph.core :as morph]
         '[tablecloth.pipeline :as tc-mm]
         '[tablecloth.api]
         '[tech.v3.ml.metamorph :as ml-mm]
         '[tech.v3.dataset.metamorph :as ds-mm]
         '[tech.v3.dataset.column-filters :as cf]
         '[camel-snake-kebab.core :as csk]
         '[clojure.string :as str]
         '[scicloj.metamorph.ml :as eval-mm]
         '[tech.v3.ml.loss :as loss]
         '[tech.v3.ml.gridsearch :as grid]
         '[tech.v3.libs.smile.classification]
         '[tech.v3.ml.classification :as classif])

["## Introduction "]

[" In this example, we will train a model which is able to predict the survival of passengers from the Titanic dataset."
 "In a real analysis, this would contain as well explorative analysis of the data, which I will skip here,
as the purpose is to showcase methamorph.ml, which is about model evaluation and selection."
 ]


["### Read data"]

(def data (tablecloth.api/dataset "data/titanic/train.csv" {:key-fn csk/->kebab-case-keyword}))


["Column names:"]
(tablecloth.api/column-names data)

["Use part of the data as new-data to predict on later."]
(def splits-1 (tablecloth.api/split->seq data :holdout {:ratio 0.1}))
(def new-data (:train (first splits-1)))

["Create a sequence of train/test used to evaluate the pipeline."]
(def splits (tablecloth.api/split->seq
             (:test (first splits-1))
             :kfold))



["### Use Pclass, Sex, title, age for prediction"]

["We want to create a new column :title which might help in the score.
This is an example of custom function, which creates a new column from existing columns,
which is a typical case of feature engineering."]

(defn name->title [dataset]
  (-> dataset
      (tablecloth.api/add-or-replace-column
       :title
       (map
        #(-> % (str/split  #"\.")
             first
             (str/split  #"\,")
             last
             str/trim)
        (data :name)))
      (tablecloth.api/drop-columns :name)))


["The pipeline definition"]

(def pipeline-fn
  (morph/pipeline
   (tc-mm/select-columns [:survived :pclass :name :sex :age])

   ;; included th custom function via lifting in the pipeline
   (morph/lift name->title)

   (ds-mm/categorical->number [:survived :pclass :sex :title])
   (ds-mm/set-inference-target :survived)

   ;; this key in the data is required by the function
   ;; scicloj.metamorph.ml/evaluate-pipeline
   ;; and need to contain the target variable (the truth)
   ;; as a dataset
   (fn [ctx]
     (assoc ctx
            :scicloj.metamorph.ml/target-ds (cf/target (:metamorph/data ctx))))

   ;; we overwrite the id, so the model function will store
   ;; it's output (the model) in the pipeline ctx under key :model
   {:metamorph/id :model}
   (ml-mm/model {:model-type :smile.classification/random-forest})
   ))

["Evaluate the (single) pipeline function using the train/test split"]
(def evaluations
  (eval-mm/evaluate-pipelines
   [pipeline-fn]
   splits
   loss/classification-accuracy
   :accuracy))


["The default k-fold splits makes 5 folds,
so we train 5 models, each having its own loss."]

["The `evaluate-pipelines` fn averages the models per pipe-fn,
and returns the best.
So we get a single model back, as we only have one pipe fn"]

["Often we consider the model with the lowest loss to be the best."]

["The evaluation results contains all data sets for test and prediction
, all models, and the loss
for each cross validation"]

["`tech.ml` stores the models ion the context in a serialzed form,
and the function `thaw-model` can be used to get the original model back.
This is a Java class in the case of
 model :smile.classification/random.forest, but this depends on the
which `model` function is in the pipeline"]

["We can get for example,  the models like this:"]

(def models
  (->> evaluations
       (map
        #(hash-map :model (tech.v3.ml/thaw-model (get-in % [:fitted-ctx :model]))
                   :mean (:mean %)
                   :fitted-ctx (:fitted-ctx %)
                   :data (get-in % [:fitted-ctx :metamorph/data])))
       (sort-by :mean)
       reverse
       ))


["The 1 (best out of 5) trained model is:"]
(map #(dissoc % :data :fitted-ctx) models)

["The one with the highest accuracy is then:"]
(:model (first models))

 ["with a accuracy of :"]

(:mean (first models))

["The pre-processed data is:"]
^kind/dataset-grid
(:data (first models))



["We can get the predictions, which for classification contain as well
the posterior probabilities per class."]

(def predictions
  (->
   (pipeline-fn
    (assoc
     (:fitted-ctx (first models))
     :metamorph/data new-data
     :metamorph/mode :transform
     ))
   :metamorph/data))

^kind/dataset
predictions

;; ["We have a helper function, which allows to predict using
;;  the best model from the result to `evaluate-pipelines`,
;; as this is a very common case."]


;; (def predictions
;;   (->
;;    (eval-mm/predict-on-best-model
;;     evaluations
;;     new-data
;;     :accuracy)))

["Out of the predictions and the truth, we can construct the
 confusion matrix."]

(def trueth
  (->
   (pipeline-fn {:metamorph/data new-data :metamorph/mode :fit })
   :metamorph/data
   tech.v3.dataset.modelling/labels))

^kind/dataset
(->
 (classif/confusion-map (:survived predictions)
                        (:survived trueth)
                        :none)
 (classif/confusion-map->ds))

["### Hyper parameter tuning"]

["This defines a pipeline with options. The options gets passed to the model function,
so become hyper-parameters of the model.

The `use-age?` options is used to make a conditional pipeline. As the use-age? variable becomes part of the grid to search in,
we tune it as well.
This is an example how pipeline-options can be grid searched in the same way then hyper-parameters of the model.

"]
(defn make-pipeline-fn [options]

  (morph/pipeline
   options
   (if (:use-age? options)
     (tc-mm/select-columns [:survived :pclass :name :sex :age])
     (tc-mm/select-columns [:survived :pclass :name :sex])
     )
   (morph/lift name->title)
   (ds-mm/categorical->number [:survived :pclass :sex :title])
   (ds-mm/set-inference-target :survived)
   (fn [ctx]
     (assoc ctx
            :scicloj.metamorph.ml/target-ds (cf/target (:metamorph/data ctx))))
   ;; any plain map in a pipeline definition get merged into the context
   ;; this is used here to configure the if for the next step
   ;; so the id can be set by the user
   {:metamorph/id :model}
   (ml-mm/model
    (merge options
           {:model-type :smile.classification/random-forest}))))

["Use sobol optimization, to find 100 grid points,
which cover in a smart way the hyper-parameter space."]

(def search-grid
  (->>
   (grid/sobol-gridsearch {:trees (grid/linear 100 500 10)
                           :split-rule (grid/categorical [:gini :entropy])
                           :max-depth (grid/linear 1 50 10 )
                           :node-size (grid/linear 1 10 10)
                           :sample-rate (grid/linear 0.1 1 10)
                           :use-age? (grid/categorical [true false])})
   (take 100))
  )

["Generate the 100 pipeline-fn we want to evaluate."]
(def pipeline-fns (map make-pipeline-fn search-grid))


["Evaluate all 100 pipelines"]
(def evaluations
  (eval-mm/evaluate-pipelines
   pipeline-fns
   splits
   loss/classification-accuracy
   :accuracy
   100
   ))

["Get the key information from the evaluations and sort by the metric function used,
 accuracy here."]

(def models
  (->> evaluations
       (map
        #(assoc
          (select-keys % [:metric :mean :fitted-ctx])
          :model (tech.v3.ml/thaw-model (get-in % [:fitted-ctx :model]))
          :data (get-in % [:fitted-ctx :metamorph/data])))
       (sort-by :mean)
       reverse))

(def best-model (first models))

["The one with the highest accuracy is then:"]
(:model best-model)

 ["with a accuracy of "  (:metric best-model)
  "and a mean accuracy of " (:mean best-model)]

["using options: "]
(-> best-model :fitted-ctx :pipeline-options)
