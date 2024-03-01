import { layers, Sequential } from "@tensorflow/tfjs";

export class Network extends Sequential {
  constructor() {
    super(
      {
        name: 'Network',
        layers: [
          layers.dense({
            units: 16,
            inputShape: [1]
          }),
          layers.dense({
            units: 1,
            inputShape: [16]
          }),
        ]
      }
    );
  }

  public compile(): void {
    super.compile({
      loss: 'meanSquaredError',
      optimizer: 'sgd'
    })
  }
}
