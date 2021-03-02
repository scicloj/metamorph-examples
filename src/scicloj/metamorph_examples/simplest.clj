(ns scicloj.metamorph-examples.simplest
  (:require [notespace.api :as notespace]
            [notespace.kinds :as kind]
            [notespace.state :as state]))


^kind/hidden
(comment
  (notespace/render-static-html)
  (notespace/init-with-browser)
  )

["# Metamorph"]

["The metamorph project establishes a simple convention to allow to
 create data transformation pipelines out of pure functions.
It allows specifically to create pipelines which are **self contained**,
 which means that they do not refer to any state (like `defs`)
outside the pipeline.
This makes the pipeline portable and re-executable with different data,
which is, for example, important for machine learning.

These constraints are explained in the [metamorph](https://github.com/scicloj/metamorph) project website.

"]

["The metamorph project does not contain any transformation function itself, but we can create them adhoc."]

["## A first pipeline"]
["In the following example, the main data object to manipulate is a simple String,
and we will create some pipeline to manipulate it." ]

["First we require the metamorph library"]
(require '[scicloj.metamorph.core :as morph]
         '[clojure.string :as str])

["Now we define a first pipeline, containing a single transform function.
It is implemented adhoc, as an anonymous function."]

["As you can see, it follows the metamorph convention,
to be a function which takes a ctx map, reads the data at key :metamorph/data
and writes it back to the ctx at the same key. (and it does not change
anything else in the ctx)"]


(def my-pipeline-fn-1
  (morph/pipeline
   (fn [ctx]
     (assoc ctx
            :metamorph/data
            (str/upper-case (:metamorph/data ctx))))

   ))

["(The following 2 forms are alternative notations of the same:)"]

^kind/hiccup-nocode
[:p/code "
   (fn [ctx]
      (update-in ctx [:metamorph/data]
                 str/upper-case))

   #(assoc % :metamorph/data
           (str/upper-case (:metamorph/data %)))
"]


["This defines a function, which can be called
with a context map:
"]

(my-pipeline-fn-1 {:metamorph/data "hello world"})


["In case we want to re-use a transformer function,
we could make a compliant function by `wrapping` the code,"]

(defn transform-uppercase []
  (fn [ctx]
     (assoc ctx
            :metamorph/data
            (str/upper-case (:metamorph/data ctx)))))


["and use it in the pipeline:"]

(def my-pipeline-fn-2
  (morph/pipeline
   (transform-uppercase)
   ))

(my-pipeline-fn-2 {:metamorph/data "hello world"})


["For defining compliant functions, we have as well a helper function,
which does the `lifting` automatically. It works for any function
which takes the main data object as first argument and returns it as well."]

["Example of lifting the clojure.core/subs function, which takes parameters:"]

(defn transform-substring [start end]
  (morph/lift subs start end))

["and using it: "]

(def upper-and-shorten-fn
  (morph/pipeline
   (transform-uppercase)
   (transform-substring 0 5)
   ))


(upper-and-shorten-fn {:metamorph/data "hello world"})


["What we have seen so far, could have been achieved as well using
the threading macro of Clojure, `->`"]

["But the `pipeline` function is not a macro,
it is a normal Clojure function and
it returns a normal Clojure function, which is again
a metamorph compliant function.
So we can compose  transformer functions and pipelines easily:"]

(def upper-and-shorten-and-lower-fn
  (morph/pipeline
   upper-and-shorten-fn
   (morph/lift str/lower-case)
   ))

(upper-and-shorten-and-lower-fn {:metamorph/data "hello world"})

["This allows to construct complex pipelines piece-by-piece and compose them,
either adhoc or externalize them in specialized libraries.
"]


["So far we have seen how to construct metamorph compliant simple pipelines.
So far the functions were independent from each other, and did not have
the need to 'communicate'"]

["This we will see in an other tutorial: "]

["https://scicloj.github.io/metamorph-examples/scicloj/metamorph-examples/deps/index.html"]
