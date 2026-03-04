const crops = [
  "Apple", "Corn", "Tomato", "Potato", "Grape",
  "Peach", "Pepper", "Strawberry", "Blueberry",
  "Soybean", "Raspberry", "Squash", "Cherry", "Orange",
];

const dataStats = [
  { value: "87,867", label: "Total Images" },
  { value: "38",     label: "Disease Classes" },
  { value: "70/15/15%", label: "Train/Val/Test" },
];

export default function About() {
  return (
    <div className="mx-auto max-w-4xl px-4 py-10 sm:px-6 sm:py-16 lg:px-8">

      {/* Page header */}
      <div className="border-b border-gray-100 pb-6 sm:pb-8">
        <h1 className="text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">
          About LeafScan AI
        </h1>
        <p className="mt-3 max-w-xl text-sm text-gray-600 sm:text-base">
          An end-to-end deep learning system for plant disease detection, built as a
          portfolio project demonstrating production ML engineering.
        </p>
      </div>

      {/* Dataset */}
      <section className="mt-10 sm:mt-12">
        <h2 className="text-lg font-semibold text-gray-900 sm:text-xl">Dataset</h2>
        <p className="mt-3 text-sm leading-relaxed text-gray-600 sm:text-base">
          Trained on approximately 87,867 RGB images of healthy and diseased crop
          leaves, categorized into 38 classes. Based on the{" "}
          <a
            href="https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset"
            target="_blank"
            rel="noopener noreferrer"
            className="text-green-600 underline underline-offset-2 hover:text-green-700"
          >
            PlantVillage dataset
          </a>
          .
        </p>

        <div className="mt-5 grid grid-cols-1 gap-4 sm:grid-cols-3">
          {dataStats.map((s) => (
            <div key={s.label} className="rounded-2xl bg-white p-6 shadow-xl sm:p-8">
              <p className="text-2xl font-bold text-green-600 sm:text-3xl">{s.value}</p>
              <p className="mt-1.5 text-sm text-gray-600">{s.label}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Crops */}
      <section className="mt-10 sm:mt-12">
        <h2 className="text-lg font-semibold text-gray-900 sm:text-xl">Supported Crops</h2>
        <p className="mt-3 text-sm text-gray-600">
          14 crop types with multiple disease states each.
        </p>
        <div className="mt-4 flex flex-wrap gap-2">
          {crops.map((crop) => (
            <span
              key={crop}
              className="rounded-full bg-white px-3 py-1.5 text-sm text-gray-700 shadow-md sm:px-4"
            >
              {crop}
            </span>
          ))}
        </div>
      </section>

      {/* Model */}
      <section className="mt-10 sm:mt-12">
        <h2 className="text-lg font-semibold text-gray-900 sm:text-xl">Model Architecture</h2>
        <div className="mt-4 rounded-2xl bg-white p-5 shadow-xl sm:p-8">
          <ul className="space-y-3 text-sm text-gray-600">
            {[
              "3× Convolutional blocks (32 → 64 → 128 filters) with MaxPooling",
              "Dropout (0.3) for regularization",
              "Dense(256, ReLU) → Dense(38, Softmax)",
              "Optimizer: Adam · Loss: Categorical cross-entropy",
              "Input: 128×128×3 RGB, normalized to [0, 1]",
              "~95% test accuracy",
            ].map((line) => (
              <li key={line} className="flex items-start gap-3">
                <span className="mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-green-100 text-xs font-bold text-green-600">
                  ✓
                </span>
                {line}
              </li>
            ))}
          </ul>
        </div>
      </section>

      {/* Author */}
      <section className="mt-10 sm:mt-12">
        <h2 className="text-lg font-semibold text-gray-900 sm:text-xl">Author</h2>
        <div className="mt-4 rounded-2xl bg-white p-5 shadow-xl sm:p-8">
          <p className="font-semibold text-gray-900">Vijaya Suhaas Nadukooru</p>
          <p className="mt-2 text-sm leading-relaxed text-gray-600">
            Built as a portfolio project to demonstrate end-to-end ML engineering — from
            data preprocessing and model training to FastAPI deployment and a modern React frontend.
          </p>
        </div>
      </section>

    </div>
  );
}
