(ns scicloj.metamorph-examples.deps
  (:require [notespace.api :as notespace]
            [notespace.kinds :as kind]
            [notespace.state :as state]))


^kind/hidden
(comment
  (notespace/render-static-html)
  (notespace/init-with-browser)
  (notespace/eval-this-notespace)
  )

["# Tranformation step with multiple outputs"]

[" Sometimes we want to add a transformation step with multiple outputs."
 "Or the shape of output is not compatible with the main data object"
 ]

["For this case, arbitrary objects can be stored in the context map."]

(require '[scicloj.metamorph.core :as morph]
         '[clojure.string :as str])


["This pipeline is an example of
 passing arbitrary data from one step to an other via the ctx."
 "The pipeline takes a string, counts the length, cuts the string to 5 characters,
counts length again and adds the 2 lengths together at the end."

 "This should show, that indeed arbitrary data processing steps can be done
and arbitrary data can be passed from a step to any following step, as long as
the steps agree on the keys to use."

 ]
(def pipe-fn
  (morph/pipeline
   (fn [ctx]
     (let [s (:metamorph/data ctx)]
       (assoc ctx
              :str-length-1 (count s)
              :metamorph/data
              (subs  (:metamorph/data ctx) 0 5))))
   (fn [ctx]
     (let [s (:metamorph/data ctx)]
       (assoc ctx
              :str-length-2 (count s))))
   (fn [ctx]
     (let [s (:metamorph/data ctx)]
       (assoc ctx
              :sum-str-length  (+ (:str-length-1 ctx)
                                  (:str-length-2 ctx)))))

   ))


(pipe-fn {:metamorph/data "hello world"})


["The resulting pipeline is still completely portable."
 "It can be stored in a variable, and called with other data."
 "All pipeline steps are pure functions, just relying on the input ctx,
so the step function can be cached easily with memoize."
 ]

(def pipe-fn
  (morph/pipeline
   (memoize
    (fn [ctx]
      (let [s (:metamorph/data ctx)]
        (assoc ctx
               :str-length-1 (count s)
               :metamorph/data
               (subs  (:metamorph/data ctx) 0 5)))))
   (memoize
    (fn [ctx]
      (Thread/sleep 5000) ;; slow on first call
      (let [s (:metamorph/data ctx)]
        (assoc ctx
               :str-length-2 (count s)))))

   (fn [ctx]
     (let [s (:metamorph/data ctx)]
       (assoc ctx
              :sum-str-length  (+ (:str-length-1 ctx)
                                  (:str-length-2 ctx)))))

   ))

["slow"]
(pipe-fn {:metamorph/data "my world"})

["fast"]
(pipe-fn {:metamorph/data "my world"})


["## Pipelines for machine learning"]


["In statistical learning, or machine learning, a special kind of data transformations appears."]


["They fall under the terms of 'fitting a model to data' and 'applying a fitted model to new data'.
of a pipeline"]



["This relates to a pipeline by the observation, that machine learning requires to run a data transformation pipeline in
2 `modes` . Once in mode `fit` to fit a model from training data , and ones in mode `transform` to transform new data using the fitted model.
. So a pipeline step should behave differently, depending on the execution `mode`."]

["Often only one step (often the last) in a longer pipeline shows this different behavior in `fit` and `transform`. All other steps, do the same in each mode.
These other steps are sometimes called `pre-processing`, while the last step is called `model` "
"In machine learning  these  2 modes are called as well train/predict."
 ]

["The metamorph pipeline supports this by standarising a context key containing the mode, namely :metamorph/mode and two
values for it: `:fit` and `transform`.
In this situation the model step wants to communicate with itself. Because it needs to pass the model trained at `:fit` to itself, via the context,
when in mode `:transform` . This is supported in `metamorph` by the `pipeline function` assigning a unique step id into the ctx (under key :metamorhp/id).
This id key can be used to store the trained model while in mode `fit` and the trained model can be retrieved under the same key while in mode `transform`.
"]

["A typicall usage is this:"]

(def my-ml-pipeline-fn
  (morph/pipeline
   ;;  pre-processing-step 1
   ;;  pre-processing-step 2
   ;;  pre-processing-step 3
   ;;  model step (this function behaves differently on :mode  = :fit  and :mode = :transform )
   ;;    In ':fit' it fits a model from the data, in 'transform' it uses the fitted model to transform new data
   )
  )

(def train-data [1 2 3])
(def test-data [4 5 6])

(def fitted-pipeline
  (my-ml-pipeline-fn {:metamorph/data train-data
                      :metamorph/mode :fit}
                       ;; :metamorph/id is set/unset by function morph/pipeline  before/after calling each pipeline step
                     ))


(def predicted-pipeline
  (my-ml-pipeline-fn
   (merge fitted-pipeline               ;; the merge makes the fitted model in the fitted ctx available for the new pipeline
          {:metamorph/data test-data
           :metamorph/mode :transform})))


["A concrete pipeline could look like below. In here, the fitted model is the mean of the data.
It can be any function applied on the data. This will often be the training of a machine learning model.
And the `transform` divides the new data by this mean to simulate the applying of
 the trained model to new data (= simulates prediction)
"]


(def my-ml-pipeline-fn
  (morph/pipeline

   (fn [ctx]
     (let [data (:metamorph/data ctx)]
       (assoc ctx
              :metamorph/data (map inc data)
              )))


   (fn [{:metamorph/keys [id data mode] :as ctx}]
     (case mode
       :fit (let [fitted-model
                  (/
                   (apply + data)
                   (count data))]
              (assoc ctx id fitted-model))
       :transform (let [fitted-model (get ctx id)
                        transformed-data (map
                                          #(/ % fitted-model)
                                          data) ]
                    (assoc ctx :metamorph/data transformed-data))))))

["In practice, you will use existing functions from libraries for each step, instead of writing them yourself, as we do here.
It's only done to show the principles."]


["This is the training and test data: "]
(def train-data [1 2 3])
(def test-data [4 5 6])


["We run the pipeline ones in mode :fit, which will call the ':fit' branch in the model function"]
(def fitted-pipeline
  (my-ml-pipeline-fn {:metamorph/data train-data
                      :metamorph/mode :fit}))


["Now the trained model is inside the fitted pipeline context.
And we can then merge the fitted ctx with new model and new data to run the pipeline in mode :transform.
This will execute the :transform branch of the model functions.
"]


(def predicted-pipeline
  (my-ml-pipeline-fn
   (merge fitted-pipeline ;; the merge makes the fitted model in the fitted ctx available for the new pipeline
          {:metamorph/data test-data
           :metamorph/mode :transform})))

predicted-pipeline
