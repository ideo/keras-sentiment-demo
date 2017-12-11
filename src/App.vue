<template>
    <section class="section">
        <div class="columns">
            <div class="column is-6 is-offset-3">

                {{modelLoadingProgress}}% : Model loading: {{modelLoading}}
                <div class="field" v-if="!modelLoading">
                    <label class="label">Message</label>
                    <div class="control">
                        <textarea class="textarea" placeholder="Enter something here and I will tell you what it means." @input="handleTextInput"></textarea>
                    </div>
                    <span v-html="sentimentValue"></span>
                </div>

    <div>
        <h3>output</h3>
<pre>
{{output}}
</pre>
    </div>

                <div class="field is-grouped">
                    <div class="control">
                        <button class="button is-link">Submit</button>
                    </div>
                </div>

            </div>
        </div>
    </section>
</template>

<script>
import axios from 'axios'
import _ from 'lodash'
import ndarray from 'ndarray'
import ops from 'ndarray-ops'

const MODEL_FILEPATH = '/static/data/twitter_seq_dense_dropout/fullModel.bin'
const ADDITIONAL_DATA_FILEPATHS = {
    wordIndex: '/static/data/twitter_seq_dense_dropout/dictionary.json',
}

const MAXLEN = 3000
const START_WORD_INDEX = 1
const OOV_WORD_INDEX = 2
const INDEX_FROM = 3

export default {
    name: 'App',
    props: {},
    data() {
        return {
            labels: ['Negative', 'Positive'],
            modelLoadingProgress: 0,
            modelLoading: false,
            modelInitializing: false,
            model: null,
            inputTextParsed: null,
            modelRunning: false,
            wordIndex: {},
            stepwiseOutput: [],
            input: new Float32Array(MAXLEN),
            output: new Float32Array(2),
        }
    },
    components: {
    },
    methods: {
        handleLoadingProgress(progress) {
            this.modelLoadingProgress = Math.round(progress)
            if (progress === 100) {
                this.modelLoading = false
            }
        },
        handleInitProgress(progress) {
            this.modelInitProgress = Math.round(progress)
            if (progress === 100) {
                this.modelInitializing = false
            }
        },
        loadAdditionalData() {
            this.modelLoading = true
            const reqs = ['wordIndex'].map(key => {
                return axios.get(ADDITIONAL_DATA_FILEPATHS[key])
            })
            axios.all(reqs).then(
                axios.spread((wordIndex) => {
                    this.wordIndex = wordIndex.data
                    this.modelLoading = false
                })
            )

        },
        handleTextInput: _.debounce(function(e) {
            if (this.modelRunning) return

            var text = e.target.value;
            if (text.trim() === '') {
                this.inputTextParsed = []
                return
            }

            this.modelRunning = true

            this.inputTextParsed = text.trim().toLowerCase().split(/[\s.,!?]+/gi)

            this.input = new Float32Array(MAXLEN)

            // by convention, use 2 as OOV word
            // reserve 'index_from' (=3 by default) characters: 0 (padding), 1 (start), 2 (OOV)
            // see https://github.com/fchollet/keras/blob/master/keras/datasets/imdb.py
            let indices = this.inputTextParsed.map(word => {
                const index = this.wordIndex[word]
                return index
            })

            let flattened32Array = this.sequences_to_matrix(indices);
            //indices = [START_WORD_INDEX].concat(indices)
            //indices = indices.slice(-MAXLEN)
            // padding and truncation (both pre sequence)
            //const start = Math.max(0, MAXLEN - indices.length)
            // for (let i = start; i < MAXLEN; i++) {
            //     this.input[i] = indices[i - start]
            // }

            for (let i = 0; i < flattened32Array.length; i++) {
                this.input[i] = flattened32Array[i]
            }

            console.log(this.input);

            this.model.predict({ input: this.input }).then(outputData => {
                this.output = new Float32Array(outputData.output)
                // this.stepwiseCalc()
                this.modelRunning = false
            })

        }, 200),

        sequences_to_matrix(sequences) {
          var oneHot = new Float32Array(3000);
          for (var i = 0; i < sequences.length; i++){
            if (sequences[i] < 3000){
              oneHot[sequences[i]] = 1.0
            }
          }
          return oneHot;
        },
        stepwiseCalc() {
            const forwardHiddenStates = this.model.modelLayersMap.get('bidirectional_1').forwardLayer.hiddenStateSequence
            const backwardHiddenStates = this.model.modelLayersMap.get('bidirectional_1').backwardLayer.hiddenStateSequence
            const forwardDim = forwardHiddenStates.tensor.shape[1]
            const backwardDim = backwardHiddenStates.tensor.shape[1]

            const start = _.findIndex(this.input, idx => idx >= INDEX_FROM)
            if (start === -1) return

            const stepwiseOutput = []
            const tempTensor = ndarray(new Float32Array(forwardDim + backwardDim), [forwardDim + backwardDim])
            for (let i = start; i < MAXLEN; i++) {
                ops.assign(tempTensor.hi(forwardDim).lo(0), forwardHiddenStates.tensor.pick(i, null))
                ops.assign(
                    tempTensor.hi(forwardDim + backwardDim).lo(forwardDim),
                    backwardHiddenStates.tensor.pick(MAXLEN - i - 1, null)
                )
                stepwiseOutput.push(this.model.layerCall('dense_1', tempTensor).tensor.data[0])
            }
            this.stepwiseOutput = stepwiseOutput
        }

    },
    computed: {
        sentimentValue() {
            var p = this.labels[0] + ' ' + (this.output[0] * 100).toFixed(1) + '%' 
            var n = this.labels[1] + ' ' + (this.output[1] * 100).toFixed(1) + '%' 
            return p + "<br>" + n
        }
    },
    watch: {
    },
    created() {
        console.log('created');
        this.model = new KerasJS.Model({
            filepath: MODEL_FILEPATH,
            gpu: false
        })
        this.model.events.on('loadingProgress', this.handleLoadingProgress)
        this.model.events.on('initProgress', this.handleInitProgress)
    },
    async mounted() {
        await this.model.ready()
        this.loadAdditionalData()
        console.log('mounted');
    }
}
</script>

<style lang="scss">
@import "~bulma";
</style>
