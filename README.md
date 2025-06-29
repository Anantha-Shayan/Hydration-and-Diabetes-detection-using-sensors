<h1>Non-Invasive Health Monitoring System for Hydration and Diabetes Risk Detection<br></h1>
<br>
<h2>Overview</h2> An interdisciplinary project that combines biosensor hardware, synthetic data generation,
machine learning, and biomedical signal analysis to build a non-invasive health monitoring system
capable of assessing skin hydration and early diabetes risk.
<br>
<h2>Objective</h2>To design and simulate a real-time health monitoring system that uses skin-based sensors
and user parameters to: - Estimate hydration levels - Assess probabilistic diabetes risk.
<br>
<h2>Motivation</h2>Diabetes is a growing global health concern and often goes undiagnosed in early stages. -
Skin hydration and thickness can serve as indirect indicators of metabolic conditions. - Wearable or
portable health tech offers promising opportunities for early screening.
<br>
<h2>Technologies & Tools</h2><h4>Sensors:</h4> <pre>*GSR (Galvanic Skin Response)<br>*Temperature Sensor (TMP36)<br>*UV Sensor(ML8511)</pre><h4>Hardware:</h4><pre>*ESP32 / Arduino</pre> <h4>Programming:</h4><pre>*Python<br>*Arduino IDE</pre>  <h5>ML Libraries:</h5><pre>scikit-learn, pandas, numpy</pre>
<br>
<h2>Project Phases (Completed):</h2>
<pre><h3>1. Ideation and Planning</h3>- Identified skin hydration and skin temperature as early indicators of diabetes related symptoms.<br>- Decided on a non-invasive, sensor-based approach for practical usability. <br>- Outlined core features: hydration detection, diabetes risk prediction, sensor fusion, user input handling.</pre>
<pre><h2>2. Feature Selection</h2>- Sensor inputs: GSR (hydration), temperature, UV index (indirect dehydration risk), skin thickness (via capacitance)
- User inputs: Age, BMI (height + weight), Pregnancy status</pre>
<pre><h2>3. Synthetic Data Generation</h2>- Defined feature ranges based on clinical norms and public datasets(e.g., PIMA)</pre>
<br>
<h2>Next Steps:</h2>
<h4>ML Model Planning (In Progress)</h4><pre>Preparing feature set for classification: [age, BMI, pregnancies,
temperature, hydration] - Training models (SVM, Random Forest) for probabilistic
output - Evaluating with accuracy, ROC AUC, confusion matrix</pre>
<h4>Hardware Simulation (Ongoing)</h4><pre>Simulating circuit with signal generators for sensor inputs</pre>
Finalize and tune ML model using synthetic data - Deploy model on ESP32 or communicate with PC/mobile for inference - Begin real sensor integration and calibration
<h2>Outcome Goal:</h2> A low-cost, user-friendly, non-invasive health screening prototype that uses multisensor fusion and machine learning to provide hydration status and diabetes risk insights
