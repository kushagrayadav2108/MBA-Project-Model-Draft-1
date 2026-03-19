
Paper Title
2025 AI-Driven Smart Healthcare for Society 5.0, February 14-15, 2025, Kolkata, India
979-8-3315-3633-6/25/$31.00 ©2025 IEEE
Energy-Efficient Protocols for Remote Patient
Monitoring Using Wireless Sensor Networks in
Telehealth Applications
(^1) Poornima N V
Symbiosis Centre for Management
Studies, Bengaluru Campus,
Symbiosis International (Deemed University),
Pune, India.
poornima@scmsbengaluru.siu.edu.in
(^4) Malik Bader Alazzam
Faculty of Information Technology,
Jadara University,
Irbid, Jordan
malikbader2@gmail.com
(^2) G Ahmed Zeeshan
Department of ECE
Global Institute of Engineering and
Technology (Autonomous),
Moinabad, Telangana, India
ahmedzeeshan.engg@gmail.com
(^5) Shamim Ahmad Khan
Department of ECE
Glocal School of Science & Technology.
Glocal University ,
U.P, India
skwarsi@hotmail.com
(^3) Sanathkumar.S
Department of Computer science
Jain University
Bangalore, India
sanathkumar1874@gmail.com
(^6) Ponni Valavan M
Department of AI & DS,
K.Ramakrishnan College of Engineering
Trichy, Tamil Nadu,India
ponnivalavankrec123@gmail.com
Abstract — A combination of wireless sensor networks in
telehealth has greatly enhanced the field of remote patient
monitoring. The concerns such as low power supply of the
sensor battery, high costs of transmitting the captured data, and
the requirement of timely identification of the health events are
still valid. He pointed out that on the conventional models the
tradeoff between monitoring in real time and energy saving is
not optimal and therefore they do not perform well in dynamic
health care buildings. The goal of this work is to build and assess
a new Energy Efficient Framework to detect health events in the
Telehealth environment. In other words, one has to maximize
the health event predictive capabilities over the health event
sensors in order to improve the quality of patient care and to
choose the right data transfer approach. The AHL-EDA
framework illustrates the proposed Adaptive Hybrid Learning
Model with Event Detection and Adaptation which are a
traditional method hybridized. The AHL-EDA framework
integrates three key components: The key HL applications of
NBs are: (1) RNNs for temporal forecasting to predict future
health events of a patient based on monitored metrics, (2) CNNs
for features that can be extracted from raw sensor signals before
transmitting large amounts of data, and (3) Reinforcement
Learning for selection or deselection of sensors based on
changing health conditions. Such a dual-layer architecture
enables the model to flexibly leverage sensors in a particular
patient’s condition and thus maximize both the benefits and
energy efficiency of patient care. The experimental results also
justify the efficiency of the AHL-EDA in the comparative
analysis with baseline systems. The AHL-EDA model
synthesized above achieved the following metrics: an accuracy
of 98%; precision of 94%; recall of 92%; and F1-score of 91%
indicating the AHL-EDA model outperformed all the baseline
models in terms of predictive accuracy and efficiency. The
results presented in this paper prove tremendous potential of the
proposed model as mean for increasing patient care by
increasing accuracy and efficiency of telehealth systems in the
future.
Keywords: Adaptive Hybrid Learning, Telehealth Monitoring,
Sensor Control Optimization, Energy Efficiency, Wireless Sensor
Networks (WSNs)

I. INTRODUCTION
Remote Patient Monitoring or RPM has been one of the
key delivery models for telehealth in primary care and
especially in chronic diseases like diabetes, hypertension, and
heart diseases [1]. through RPM, health status can be
continuously monitored, and then intervention can be
provided remotely. The continuous delivery of health care is
made possible by WSNs through the gathering and relay of
physiological data from portable or invasive sensors and
forwarding of data to central controlling units [2]. These
networks involve indirect data monitoring and they are useful
in identification of early signs of detoriation that need
intervention. Nonetheless, decentralized RPM systems ‘re
usability remains a matter of the reliability of the WSNs that
support them [3]. As aforementioned, one of the major
problems of WSNs in RPM applications is energy concerns.
As is explained in the following, most of the sensors that are
deployed in WSNs are powered by batteries, and these
batteries need to be capable of powering the sensors for a long
period of time, sometimes measuring in terms of months or
even years in some cases, without the need for human
intervention [4]. Prolonged data monitoring of patient health
information consumes a lot of energy in sensors, which may
lead to frequent replacement or recharging [5]. This also
drives up operating expenses because a compromised network
life weakens the efficiency of the sensor nodes that need
timely power to identify severe health conditions [6]. This
becomes even worse in real time health monitoring where
envisaging quick data transmission and activation of the
sensors goes hand in hand with the overall welfare of the
patients [7].
To overcome these challenges, the proposed AHL-EDA,
Adaptive Hybrid Learning based Energy Driven Architecture,
combines modern machine learning techniques additionally
with reinforcement learning to enhance the lifetime of WSNs
while using the sensors effectively. In the case of fluctuating
conditions within a patient, AHL-EDA can switch on sensors
as appropriate and turn off the data transmission, if not
required. The model: saves battery power, reduces the need
for sensor replacement, and optimizes the general
effectiveness of the monitoring system by decreasing the
number of active sensors. This is a significant addition to the
longer term chick of rolling out sustainable and affordable
remote patient monitoring systems that will guarantee the
continuous care and support of patients once they are
diagnosed with the virus without compromising the stability
of the network in the process.

The AHL-EDA model reduces energy
consumption in WSN for RPM by almost 20
times compared to traditional system; sensor life
and network strength are also boosted.
AHL-EDA has better results in tuning in to
important health events like arrhythmia,
hypertensive crisis with lower latency and higher
accuracy than the traditional models.
AHL-EDA allows individual sensors to be turned
on or off as a result of patient status, thereby
avoiding excess data transmission and power
consumption without compromising monitoring
efficiency.
II. LITERATURE REVIEW
Telemedicine is a subset of telehealth, RPM has proved
efficient in chronic illnesses and especially in the management
of patients with diabetes, hypertension, heart diseases and the
elderly. However, it is still an issue of finding good
approaches to making this kind of monitoring happen in
appropriate time and with the desired quality and energy
efficiency, especially when considering massive and never-
ending flow of sensor data [8]. Wireless Sensor Networks
(WSNs) are crucial in Remote Patient Monitoring (RPM)
systems to monitor vital signs (pulse rate, blood pressure) [9].
In most cases, WSNs are made up of many sensors that are
connected and send information to a control system via the air.
However, these networks have power constraints which are
acute for battery driven sensors that must remain active for a
longer time [10]. Traditional and state-of-the-art techniques in
WSNs used in healthcare by ML and DL models are SL,
RNNs, CNNs were discussed earlier [11]. It needs to be
mentioned that RNNs are especially efficient in analyzing
time-series data which allows for predicting the likelihood of
the health events such as cardiac arrhythmias or epileptic
seizure [12]. CNNs, on the other hand, demonstrate feature
learning from sensor signals for improved identification and
categorization of health events [13]. These protocols are
designed to reduce power consumption through switching
on/off sensors whose operation is proportional to some
threshold or patient status. Although these protocols explored
could help to minimize power consumption, they are not well
equipped to handle real-time event recognition and system
stability in such conditions [14]. Currently, there is a lack of
appropriate mechanisms for enabling the sensors to activate
based on the relative complexity and variability associated
with data related to patient health.

A. Gaps in Current Research

There are several limitations to the current work in RPM
and WSNs that may deserve further attention. Current models
fail at times to optimally minimize energy consumption versus

optimizing the identification of health events [15]. Moreover,
existing approaches do not allow for easily adjusting for the
fluctuations inherent to the state of patients. These gaps are
plugged by the AHL-EDA model that uses machine learning
techniques that include recurrent neural networks and
convolutional neural networks coupled with principles of
reinforcement learning to design a sensor management and
event detection system which is dynamic in nature. This new
approach of using stationary and energy-efficient monitors for
health event prediction represents a novel contribution to the
assessment of telehealth monitoring systems.
III. RESEARCH FRAMEWORK
The suggested AHL-EDA model improves the remote
patient monitoring via WSN with the help of adaptive hybrid
learning model. AHL-EDA utilizes Recurrent Neural
Networks (RNN) for temporal health event prediction,
Convolutional Neural Network (CNN) for feature learning,
and Reinforcement Learning (RL) for triggering sensor
actuation. This integration enables AHL-EDA to track the
happening of health events comprehensively while utilizing
energy efficiency through triggering sensors sporadically.
Cuts data transference time and raise detection precision make
AHL-EDA a precise and effective method for close to real-
time telehealth surveillance.
Fig. 1. Workflow of the suggested Approach
A. Data Collection
The MIMIC-III, and MIMIC-IV sources are useful in
gathering essential physiological data of the patients required
in making a model in health monitoring. These public,
deidentified databases, which provide abundant data about
ICU patients, encompass waveform data together with
patient’s vitals, ECG, laboratory results, and clinical markers.
To use this data most efficiently, researchers can integrate a
collection flow model based on the steps of signal
identification and selection: Physiological signals are
specifically selected from an array of Health signals and
encompass heart rate, blood pressure, and respiratory rate
pertinent to early detection of possible health events [16].
B. Data Pre-processing

The normalization of data and data scaling are critical
stages when preparing data for machine learning models,
especially for those getting needed input in the form of neural
networks or other distance-based models. Standardization
refers to the bodying of all the data at an interval of 0 to 1.
This is done along with the feature scaling in which every
feature is scaled between 0 and 1 by Min-Max scaling
technique so that no feature will dominate the others. The
Min-max normalization wad expressed in (1):

=

  
^ (1)
Here,

 - The normalized value of 

 - The original value to be normalized.

 - The minimum value in the dataset



The maximum value in the dataset
C. CNN for Feature Extraction

Convolutional Neural Networks commonly work for
feature extraction of structured data types like Images or
signals collected from the sensors. In this closely related and
integrated hybrid model, CNNs perform feature extraction on
time-series sensor data such as ECG, blood pressure, and heart
rate data and then minimize their dimensionality. CNN takes
spatial features and gives out high-level features of an image,
which is then processed by RNN for temporal features. The
CNN works in a manner that a set of convolutional filters is

applied to the input signal. The convolution operation for a

1D time-series signal can be written as in (2):

= ∗ +  (2)
Here,

The input value at time step
 - The weight or coefficient associated with the input.

 - The bias term, which is added to adjust the output

Every convolutional layer convolves a number of filters to
identify such features as increases, decreases, or plain in the
given signal. The max pooling layers are then used to down
sample the data in order to retain only the last pertinent
features while pulling out the heavy computing as well as data
transfer load in applications such as real-time. The output
from the last convolutional layer is then flatten and feed into
the RNN for sequence analysis.

D. RNN for Predictive Modeling

Recurrent Neural Networks (RNNs) are very famous in
handling sequential data due to their capability that it can
maintain hidden states that contain information about past
inputs. The context of healthcare, data like patient’s heart rate,
blood pressure and ECG signals are the time series data which
is suitable to RNN’s. The unique selling point of RNN is its

aptitude to learn long-term dependencies in this sequential
data which helps it to predict future health events from the past
occurrence of health markers in a patients’ record. The basic
RNN model uses the following recurrence relation to update
its hidden state at each time step in (3):
ℎ = ℎℎ−1+  + ℎ (3)
 - The activation function applied to the computed value.
ℎ - The weight matrix for the previous hidden state
ℎ−1 - The hidden state from the previous time step.
 - The weight matrix for the current input
The input at time
ℎ - The bias term for the hidden state
This hidden state ℎ is then passed through a fully
connected layer to give predictions for health event, for
instances; the probability of heart attack or a seizure in the
next few hours. The output  is computed as:
 = ℎ +  (4)
 - The weight matrix for mapping the hidden state to the
output
ℎ - The hidden state at time
 - The bias term for the output
The model operates on the patterns of intermediate patient
data sequences, which in fact makes it possible to accurately
predict further health events.
E. Reinforcement Learning for Sensor Control
Reinforcement Learning (RL) is incorporated into this
model to attain an optimal sensor activation policy that helps
the system dynamically schedule the use of the sensors based
on the dynamic condition of the patient. RL agent’s objective
is therefore to determine when it is optimal to turn on or off
the sensor, depending on the expected reward some of which
may include patient state, battery capacity and data
transmission. This is common in the RL framework where an
agent interacts with an environment in order to learn through
a reward signal. In the context of the state within the
healthcare environment, certain information can be noted,
namely, the patient’s basic biomedical parameters, the status
of the sensors: active or inactive, battery charge. The action
can be decisions like on or off a sensor, or simply a change in
the rate at which it collects data. The agent’s goal is to fully
optimize the reward; this notion has components based on the
immediate experience and the predictions about the distant
future. The reward function may involve inflicting a penalty
for over transmitting data or getting positive consequences for
accurately predicting health events. The reward at time step
can be written as in (5):
 =  ,  (5)
 - The reward function that determines the reward based on
the state and action
 - The state of the system or environment at time
The action taken by the agent at time
The Q learning algorithm, is one of the most used RL
algorithm is used to update the Q values corresponding to state
action space. The Q-value represents the expected future
reward for a given state-action pair and is updated as in (6):

 , ←  , + " # + γmax
()
 *+, )−
 , , (6)
 ,  - The current estimate of the action-value (Q-

value) for state  and action
" - The learning rate, which controls how much the Q-value

is updated in each iteration

 - The immediate reward received after taking action state
 and action
Γ - The discount factor, which determines the importance of

future rewards

max
′

/+1, ′ 0 - The maximum Q-value for the next state
+1 across all possible actions )
+1 - The state of the system after taking action
) - A potential action in the next state

The RL agent explores the environment (using random
actions), and then exploits known high-reward actions in order
to find the best policy for sensor activation. This not only
minimizes wasteful movement of data around the system but
also assures that significant patient events are real time
recording. The end product is therefore, an effective resource
utilization system in the overall health monitoring of the civil
aviation system.

F. Integration of AHL-EDA Architecture

The proposed AHL-EDA has three effective modules:
RNN, CNN, and RL for real-time health monitoring in the
context of telehealth applications. The RNN component aims
at providing an ability to predict health events over time by
processing, rhythmical biomarkers temporality of the patient,
including, for example, heart rate, pressure of blood,
respiratory rate. It retains the order in the data necessary for
model’s ability to forecast the occurrence of subsequent health
events, such as arrhythmias or hypertensive crises, based on
the patient’s physiological data signals. The CNN component
is used to enable feeding the high-density sensor data; for
instance, ECG or other sensors into the model in an efficient
manner. In this case, it offers feature extraction which
involves selecting important patterns in the data without
having to send all the raw data from the client node to the base
node or even from one cluster head to another. It also reduces
the overall computational load and the transmission
bandwidth hence making the system power friendly. Through
the CNNs, the model is able to discard the surplus features that
might overload a system and does not provide important
features for accurate event detection.

The RL component that considers the adaptive control of
sensors mean that the usage of a certain sensor is carefully
preconditioned on the basis of the condition of the patient. The
RL agent is trained to determine whether and when to activate,
deactivate or sample the sensors at a lower rate to find the
mean balance between the amount of data required to
recognize health events efficiently and energy consumption as
well as data transmission. The integration of the RNN, CNN,
and RL together develops an adaptive and elastic systems that
can learn from a patient’s changing condition. This overall
integration enables the AHL-EDA architecture work to
facilitate the large amount of data it processes and the decision
making that sustains efficient sensor operation. The system
guarantees that possible aggregate health events are identified
early enough; can adjust quickly to variations in the health
states of the patient; and lastly imposes minimal operating
needs in terms of on demand assets to existing health care
structures as well as on the specific wearable devices readily
available to the patient. Adding temporal forecasting, optimal
feature extraction, and sensor control makes the presented
model reliable and energy-saving for critical telehealth
applications that need real-time measurement and limited
power consumption.
IV. RESULT AND DISCUSSION
The section quantitatively assesses the AHL-EDA model
deploying outcome criteria such as accuracy, precision, and
recall together with F1-score, the rate of sensors’ activation,
data transmission, energy consumption, as well as real-time
detection latency. These measurements portray the
performance of the developed model in identifying the health
events, managing energy consumption, and implementing
efficient communication. The analysis of AHL-EDA
performance in the context of other models such as RNN,
CNN, and traditional approaches confirms its higher accuracy
in energy efficiency. The presented framework is a prototype
implemented in a Python simulation tool.
A. Sensor Activation and Resource Efficiency Comparison
Fig. 2 compares sensor activation rate, data transmission,
and energy consumption across models: AHL-EDA, RNN,
CNN, and a traditional approach are used for the identification
of the signals. The proposed AHL-EDA provides the best
performance with 62 percent sensor activation rate, 85 MB
data transmission, and 150 Joules energy consumption while
results in lower values as compared to other baseline models
that presents high resource consumption. This underscores the
monitoring trade-off between accuracy and energy in the
systems of AHL-EDA adaptive telehealth: Prolonging the
useful lifecycle of health monitoring systems whilst ensuring
that health events are recognized in real time.
Fig. 2. Resource Efficiency Assessment
B. Real-Time Detection Latency for Health Events

Fig. 3 presents the difference in detection latency when
using the AHL-EDA model vs. the baseline models (RNN,
CNN, and the traditional model) at different CHiE. As shown,
regardless of the event, AHL-EDA model exhibits time
latency of approximately 1.0 – 1.3 seconds for detections. For
instance, it identifies arrhythmia at a time of 1.2sec which is
faster than RNN 1.5sec, CNN 1.4sec, and the traditional
2.0sec. Comparable trends are revealed for hypertensive crisis,
sepsis, and respiratory failure as well. These results show that
AHL-EDA outperforms other methods in real-time health
event detection in a significantly faster manner which will
prove very useful in monitoring patients in telehealth systems.

Fig. 3. Real-time Detection Latency Analysis
C. Model Assessment

Evaluation of predictive models requires the use of
performance metrics. Table I provides some basic assessments
of the AHL-EDA in contrast to four compared baseline
models: RNN, CNN, SVM, and Logistic Regression. Four
measures, Accuracy, Precision, Recall, and F1-Score, to
compare the performance of each model to pick health events
from the sensor data in the telehealth context and show AHL-
EDA provides better results.

TABLE I. PERFORMANCE METRICS ASSESSMENT
Model Accuracy
(%)
Precision
(%)
Recall
(%)
F1-Score
(%)
Proposed
AHL-EDA
98% 94% 92% 91%
RNN 89% 86% 88% 87%
CNN 86% 84% 85% 84.5%
SVM 82% 80% 83% 81.5%
Logistic
Regression
78% 75% 77% 76%
The AHL-EDA model with natures of RNNs, CNNs and
Reinforcement Learning outperforms all parameters with an
impressive accuracy of 98 % as against other baseline models.
They have obtained a high accuracy of 94%, thus providing a
robust prediction of the health event, recall of 92% in
capturing actual events and F1 score of 91% showing a
balanced accuracy and reliability for telehealth monitoring.
AHL-EDA also entails a better performance over the baseline
models to interpret the healthcare sensor data with a higher
accuracy of 98% than RNN, 89% and CNN 86%, SVM and
Logistic regression 82% and 78% respectively. This proves
that AHL-EDA is self-governing, comprehensive, sensitive

and efficient in perceiving health events with adaptive control
and intelligent usage of resources. These findings show that
ACL-EDA has high capabilities for energy efficient, real-time,
monitoring in telecare than the conventional methods.
Fig. 4. Comparative assessment
V. CONCLUSION AND FUTURE SCOPE
In this study, the AHL-EDA model unveiled by the author
demonstrates comprehensive benefits for increasing energy
efficiency and adaptable watchfulness in RPM systems. It was
proposed AHL-EDA model outcompeted other models in
terms of accuracy, with a score of 98% attained. Such high
accuracy is the evidence of good estimation of the health
events and reasonable control of the sensors activation and
data transmission. Through the real-time patient data, AHL-
EDA besides guaranteeing the proper identification of the
health events, constantly control sensor’s energy usage and
thus the batteries’ discharge, enhancing the energy
management and demising sensor utilization. This accuracy
value demonstrates the stability of the model, and therefore it
is more preferable for real-time low-power remote patient
monitoring (RPM) applications. AHL-EDA added more
benefits to patients as it performed better than most traditional
methods and the baseline models because its approach of
changing the senor activation depending on the current patient
status helped to save energy and enhance the lifespan of the
sensors. These results highlight the applicability of AHL-EDA
as a solution to the main issues of energy deficits in WSNs for
telehealth. As for the implications, it proposed that possibly,
the AHL-EDA model could be reinforced by retraining it
periodically or calibrating it from time to time, in a bid to bring
it back to optimal performance.
REFERENCES
[1] A. Abbas and I. Torshin, “Telemedicine and Remote Patient
Monitoring: The Future of Healthcare Delivery,” Nov. 2023. doi:
10.13140/RG.2.2.15579.13609.
[2] B. M. Mahmmod et al. , “Patient Monitoring System Based on Internet
of Things: A Review and Related Challenges With Open Research
Issues,” IEEE Access , vol. 12, pp. 132444–132479, 2024, doi:
10.1109/ACCESS.2024.3455900.
[3] E. E. Thomas et al. , “Factors influencing the effectiveness of remote
patient monitoring interventions: a realist review,” BMJ Open , vol. 11,
no. 8, p. e051844, Aug. 2021, doi: 10.1136/bmjopen-2021-051844.
[4] M. Bathre and P. K. Das, “Smart dual battery management system for
expanding lifespan of wireless sensor node,” International Journal of
Communication Systems , vol. 36, no. 3, p. e5389, 2023.
[5] Z. Gao et al. , “Advanced energy harvesters and energy storage for
powering wearable and implantable medical devices,” Advanced
Materials , vol. 36, no. 42, p. 2404492, 2024.
[6] G. K. Ijemaru, K. L.-M. Ang, and J. K. Seng, “Wireless power transfer
and energy harvesting in distributed sensor networks: Survey,
opportunities, and challenges,” International Journal of Distributed
Sensor Networks , vol. 18, no. 3, p. 155014772110677, Mar. 2022, doi:
10.1177/15501477211067740.
[7] S. Abdulmalek et al. , “IoT-Based Healthcare-Monitoring System
towards Improving Quality of Life: A Review,” Healthcare , vol. 10,
no. 10, p. 1993, Oct. 2022, doi: 10.3390/healthcare10101993.
[8] A. Haleem, M. Javaid, R. P. Singh, and R. Suman, “Telemedicine for
healthcare: Capabilities, features, barriers, and applications,” Sensors
International , vol. 2, p. 100117, Jul. 2021, doi:
10.1016/j.sintl.2021.100117.
[9] “Wireless Body Sensor Networks for Real-Time Healthcare
Monitoring: A Cost-Effective and Energy-Efficient Approach,” J
Angiotherapy , vol. 8, no. 7, pp. 1–13, Jul. 2024, doi:
10.25163/angiotherapy.879796.
[10] F. Conceicao, F. B. Teixeira, L. M. Pessoa, and S. Robitzsch, “Energy-
Efficiency Architectural Enhancements for Sensing-Enabled Mobile
Networks,” Oct. 25, 2024, arXiv : arXiv:2410.19589. Accessed: Nov.
11, 2024. [Online]. Available: http://arxiv.org/abs/2410.
[11] J. Yu, A. de Antonio, and E. Villalba-Mora, “Deep Learning (CNN,
RNN) Applications for Smart Homes: A Systematic Review,”

Computers , vol. 11, no. 2, Art. no. 2, Feb. 2022, doi:
10.3390/computers11020026.
[12] A. Dabas, “Application of Recurrent Neural Networks (RNNs) in
Medical Diagnostics,” Aug. 2024. Accessed: Nov. 11, 2024. [Online].
Available: https://hal.science/hal-
[13] M. Zaher, A. S. Ghoneim, L. Abdelhamid, and A. Atia, “Unlocking the
potential of RNN and CNN models for accurate rehabilitation exercise
classification on multi-datasets,” Multimed Tools Appl , Apr. 2024, doi:
10.1007/s11042-024-19092-0.
[14] Z. Ma, Z. Hao, and Z. Zhao, “An investigation on energy-saving
scheduling algorithm of wireless monitoring sensors in oil and gas
pipeline networks,” Energy Informatics , vol. 7, no. 1, p. 104, Oct. 2024,
doi: 10.1186/s42162-024-00412-5.
[15] V. Nkemeni et al. , “Evaluation of Green Strategies for Prolonging the
Lifespan of Linear Wireless Sensor Networks,” Sensors , vol. 24, no.
21, Art. no. 21, Jan. 2024, doi: 10.3390/s24217024.
[16] A. E. W. Johnson et al. , “MIMIC-IV, a freely accessible electronic
health record dataset,” Sci Data , vol. 10, no. 1, p. 1, Jan. 2023, doi:
10.1038/s41597-022-01899-x.
This is a offline tool, your data stays locally and is not send to any server!
Feedback & Bug Reports