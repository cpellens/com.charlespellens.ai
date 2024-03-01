<script lang="ts">
  import { Network } from "$lib";
  import { setBackend, loadLayersModel } from "@tensorflow/tfjs";
  import { Tensor, tensor2d } from "@tensorflow/tfjs";
  import { onMount } from "svelte";
  import {
    Button,
    ButtonSet,
    CodeSnippet,
    FormGroup,
    NumberInput,
    Tab,
    TabContent,
    Tabs,
    TextInput,
  } from "carbon-components-svelte";
  import "carbon-components-svelte/css/white.css";

  let iterations: number = 25;
  let goalPercentage: number = 95;
  let goalHitCountTarget: number = 5;

  let network: Network;
  let input: string;
  let output: string;
  let accuracy: number;

  let trainingData = {
    inputs: [0],
    outputs: [0],
    disabled: false,
  };

  async function load(): Promise<void> {
    await setBackend("webgl");

    try {
      network = (await loadLayersModel("localstorage://my-model")) as Network;
    } catch {
      network = new Network();
    }

    network.compile();

    console.log("loaded model");
  }

  async function trainValue(input: Tensor, output: Tensor): Promise<void> {
    const targetAccuracy: number = goalPercentage * 0.01;

    accuracy = 0;

    let goalMetCount: number = 0;
    let isTraining: boolean = true;

    do {
      await network.fit(input, output, { epochs: 100 });

      /**
       * Phase 2: Make predictions
       */
      let countCorrect = 0;
      let answers = [];

      for (let i = 0; i < trainingData.inputs.length; i++) {
        const prediction = network.predict(
          tensor2d([trainingData.inputs[i]], [1, 1], "float32")
        ) as Tensor;
        const predictionOutput = Array.from(prediction.dataSync()).shift() || 0;
        const expectedOutput = trainingData.outputs[i];

        const isCorrect =
          Math.round(10 * expectedOutput) === Math.round(10 * predictionOutput);

        answers.push({ input, expectedOutput, predictionOutput });

        isCorrect && countCorrect++;
        accuracy = countCorrect / trainingData.inputs.length;
      }

      const isGoalMet = accuracy >= targetAccuracy;
      isGoalMet ? goalMetCount++ : (goalMetCount = 0);

      const isCountMet = goalMetCount >= goalHitCountTarget;

      isTraining = !isCountMet;

      console.log(
        `accuracy: ${countCorrect} of ${iterations} = ${accuracy * 100}%`,
        answers
      );
    } while (isTraining);
  }

  async function train(): Promise<void> {
    const xs = tensor2d(
      trainingData.inputs.map((e) => [e]),
      [trainingData.inputs.length, 1],
      "float32"
    );
    const ys = tensor2d(
      trainingData.outputs.map((e) => [e]),
      [trainingData.outputs.length, 1],
      "float32"
    );

    trainingData.disabled = true;
    network.trainable = true;
    await trainValue(xs, ys);
    trainingData = {
      inputs: [0],
      outputs: [0],
      disabled: false,
    };
  }

  function addInput(): void {
    trainingData.inputs.push(0);
    trainingData.outputs.push(0);

    trainingData = trainingData;
  }

  async function predict(): Promise<any> {
    const inputValue = parseInt(input);
    if (!inputValue) {
      output = "";
    } else {
      const prediction = network.predict(
        tensor2d([parseInt(input)], [1, 1])
      ) as Tensor;
      output =
        Array.from(prediction.round().dataSync()).shift()?.toString() || "";
    }

    return output;
  }

  async function save(): Promise<void> {
    await network.save("localstorage://my-model");
    console.log("saved model");
  }

  async function reset(): Promise<void> {
    localStorage.clear();
    console.log("cleared");

    await load();
  }

  onMount(load);
</script>

<Tabs type="container">
  <Tab label="Parameters" />
  <Tab label="Train" />
  <Tab label="Predict" />

  <svelte:fragment slot="content">
    <TabContent>
      <FormGroup>
        <NumberInput
          label="Validation Iteration Count"
          helperText="How many predictions should be made during each iteration?"
          bind:value="{iterations}"
        />
      </FormGroup>

      <FormGroup>
        <NumberInput
          label="Target Accuracy Percentage"
          helperText="What percentage of the training predictions need to be correct in order to complete?"
          bind:value="{goalPercentage}"
        />
      </FormGroup>

      <FormGroup>
        <NumberInput
          label="Goal Validation Count"
          helperText="How many times does the accuracy percentage goal need to be met in succession?"
          bind:value="{goalHitCountTarget}"
        />
      </FormGroup>

      <Button
        on:click="{reset}"
        kind="danger">Reset Model</Button
      >
    </TabContent>

    <TabContent>
      {#each trainingData.inputs as i, index}
        <FormGroup legendText="{`Set #${index + 1}`}">
          <NumberInput
            label="Input"
            bind:value="{trainingData.inputs[index]}"
          />
          <NumberInput
            label="Expected Output"
            bind:value="{trainingData.outputs[index]}"
          />
        </FormGroup>
      {/each}
      <ButtonSet>
        <Button
          on:click="{addInput}"
          kind="secondary">Add Input</Button
        >
        <Button
          on:click="{train}"
          bind:disabled="{trainingData.disabled}"
        >
          {trainingData.disabled
            ? `${Math.floor(0.5 + accuracy * 100)}%`
            : "Train"}
        </Button>
      </ButtonSet>
    </TabContent>

    <TabContent>
      <FormGroup>
        <TextInput
          bind:value="{input}"
          on:keyup="{predict}"
          labelText="Input"
        />
      </FormGroup>

      {#if output}
        <FormGroup legendText="Prediction Output">
          <CodeSnippet>{output}</CodeSnippet>
        </FormGroup>
      {/if}

      <ButtonSet>
        <Button on:click="{save}">Save Model</Button>
      </ButtonSet>
    </TabContent>
  </svelte:fragment>
</Tabs>
