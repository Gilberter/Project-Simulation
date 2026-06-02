# Sim-to-Real Noise Module for Transient Rendering

## Abstract

Transient rendering systems such as **Mitsuba** and **Mitransient** generate physically accurate simulations of light transport over time. However, the resulting transient measurements are often idealized and noise-free, whereas real-world sensors such as **Single-Photon Avalanche Diodes (SPADs)** and **LiDAR** systems are affected by multiple sources of uncertainty and noise.

The objective of this project is to reduce the **simulation-to-reality (Sim-to-Real) gap** by introducing a physically motivated sensor noise model for transient renders. The proposed module simulates key phenomena observed in real photon-counting devices, including **Shot Noise**, **Dark Count Rate (DCR)**, and the **Instrument Response Function (IRF)**. The resulting framework generates transient measurements that more closely resemble data acquired by real-world time-of-flight imaging systems.

---

# Introduction

## Transient Rendering

Traditional rendering (also called **steady-state rendering**) assumes that light propagates instantaneously throughout a scene. As a result, only the final radiance arriving at each pixel is computed.

Transient rendering introduces a temporal dimension by accounting for the finite speed of light. Instead of only measuring radiance, the renderer records **when photons arrive** at the sensor.

A transient image can therefore be represented as

[
I(x,y,t)
]

where:

* (x,y) denote pixel coordinates,
* (t) represents time after the emission of a laser pulse.

This enables visualization of light propagation as it travels through the scene and interacts with objects.

---

## Time-of-Flight (ToF)

Transient rendering is closely related to **Time-of-Flight (ToF)** sensing.

A laser pulse is emitted into the scene, reflected by objects, and eventually detected by a sensor. By measuring the travel time of photons, the distance to objects can be estimated.

The distance is computed as

[
d = \frac{ct}{2}
]

where:

* (d) is the distance,
* (c) is the speed of light,
* (t) is the measured round-trip travel time.

For example, if a transient peak appears at

[
t = 10\ \text{ns}
]

then

[
d = \frac{(3\times10^8)(10\times10^{-9})}{2}
]

[
d = 1.5\ \text{m}
]

---

# Motivation

Mitransient produces highly accurate simulations of light transport. However, the resulting transient measurements are effectively perfect:

* No photon counting uncertainty.
* No detector jitter.
* No dark counts.
* No laser pulse broadening.

Real SPAD sensors exhibit all of these effects.

Consequently, machine learning models trained on idealized transient data often perform poorly when deployed on real hardware.

To address this issue, we propose a **Sim-to-Real Noise Module** capable of transforming ideal transient renders into realistic sensor measurements.

---

# Proposed Contributions

The project contributes a physically inspired sensor model composed of three major components:

1. Shot Noise
2. Dark Count Rate (DCR)
3. Instrument Response Function (IRF)

The implementation operates as a post-processing stage applied directly to transient render outputs.

---

# Photon Transport Model

The transient signal produced by a renderer can be expressed as

[
I(x,y,t)
========

\sum_{p}
L(p)
,
\delta
\left(
t-\frac{\ell(p)}{c}
\right)
]

where:

* (L(p)) is the radiance transported along path (p),
* (\ell(p)) is the path length,
* (c) is the speed of light,
* (\delta) is the Dirac delta function.

This represents an idealized measurement before sensor imperfections are introduced.

---

# Shot Noise

## Physical Interpretation

Photons arrive at the detector independently and randomly.

Even if the illumination intensity remains constant, the number of detected photons fluctuates due to the quantum nature of light.

Photon arrivals are modeled using a **Poisson distribution**.

---

## Mathematical Model

For a transient bin with expected photon count (\lambda),

[
K \sim \text{Poisson}(\lambda)
]

with probability

[
P(K=k)
======

\frac{\lambda^k e^{-\lambda}}
{k!}
]

where:

* (K) is the observed photon count,
* (\lambda) is the expected photon count.

The Poisson distribution has

[
\mu = \lambda
]

and

[
\sigma^2 = \lambda
]

which implies

[
\sigma = \sqrt{\lambda}
]

The signal-to-noise ratio becomes

[
\text{SNR}
==========

# \frac{\lambda}{\sqrt{\lambda}}

\sqrt{\lambda}
]

Thus, collecting more photons naturally improves measurement quality.

---

# Dark Count Rate (DCR)

## Physical Interpretation

A SPAD can trigger even when no photon arrives.

These false detections are caused by:

* Thermal generation of carriers,
* Tunneling effects,
* Semiconductor imperfections.

As a result, the sensor records a baseline level of noise even in complete darkness.

---

## Mathematical Model

Dark counts are also modeled as a Poisson process:

[
D \sim \text{Poisson}(\lambda_{dark})
]

The observed photon count becomes

[
\lambda_{total}
===============

\lambda_{signal}
+
\lambda_{dark}
]

and

[
K
\sim
\text{Poisson}
(
\lambda_{total}
)
]

This creates a uniform noise floor across the transient histogram.

---

# Instrument Response Function (IRF)

## Physical Interpretation

Real sensors do not respond instantaneously.

Several effects broaden the measured signal:

* Finite laser pulse width,
* SPAD timing jitter,
* Electronic timing uncertainty,
* Cable and circuit delays.

Therefore, an ideal photon arrival is recorded as a temporally blurred pulse.

---

## Linear System Model

The sensor can be modeled as a linear time-invariant system.

The measured signal is given by

[
y(t)
====

(I * h)(t)
]

where:

* (I(t)) is the ideal transient signal,
* (h(t)) is the Instrument Response Function,
* (*) denotes convolution.

Expanding the convolution,

[
y(t)
====

\int_{-\infty}^{\infty}
I(\tau)
h(t-\tau)
,d\tau
]

---

## Gaussian IRF

A common model is the Gaussian kernel

[
h(t)
====

\frac{1}
{\sigma \sqrt{2\pi}}
\exp
\left(
-\frac{t^2}
{2\sigma^2}
\right)
]

where (\sigma) controls the temporal spread.

This model represents symmetric timing uncertainty.

---

## Skewed Gaussian IRF

Real SPAD systems often exhibit asymmetric responses.

A skewed Gaussian better captures:

* Fast rise times,
* Slow decays,
* Afterpulsing effects.

This model more closely matches real sensor measurements.

---

# SPAD Sensor Model Pipeline

The proposed simulation pipeline follows four stages:

### 1. Ideal Transient Render

[
I(x,y,t)
]

Generated by Mitransient.

### 2. IRF Convolution

[
I_{irf}
=======

I * h
]

Simulates temporal blur.

### 3. Shot Noise

[
I_{shot}
\sim
\text{Poisson}
(I_{irf})
]

Simulates photon-counting uncertainty.

### 4. Dark Count Rate

[
I_{final}
=========

I_{shot}
+
D
]

Produces realistic SPAD measurements.

---

# Experimental Evaluation

## Experiment 1: Noise vs Averaging

### Objective

Evaluate how repeated laser shots improve reconstruction quality.

### Method

Generate a noisy transient measurement and average:

* 10 shots
* 100 shots
* 1000 shots

### Expected Result

The signal-to-noise ratio should improve approximately as

[
\text{SNR}
\propto
\sqrt{N}
]

where (N) is the number of laser repetitions.

---

## Experiment 2: Hidden Object Color

### Question

Can a red laser detect a blue object?

### Method

Compare:

* White object
* Red object
* Blue object

using a red laser pulse.

### Expected Result

The blue object absorbs most red photons, producing a much weaker transient response that may disappear beneath the noise floor.

---

## Experiment 3: Relay Wall Material

### Question

Does the relay wall material affect transient measurements?

### Method

Compare:

* Diffuse wall
* Rough plastic wall

while keeping the scene unchanged.

### Expected Result

The glossy wall generates a strong early transient peak that can obscure weaker returns from hidden objects.

---

# Applications

The proposed Sim-to-Real module is useful for:

* Time-of-Flight imaging
* SPAD sensor simulation
* LiDAR simulation
* Non-Line-of-Sight (NLOS) imaging
* Machine learning dataset generation
* Benchmarking transient reconstruction algorithms

---

# Conclusion

This project introduces a physically motivated Sim-to-Real Noise Module for Mitransient. By incorporating Shot Noise, Dark Count Rate, and Instrument Response Functions, the framework produces realistic transient measurements that better approximate real SPAD sensors.

The proposed contribution is lightweight, easy to integrate, and scientifically meaningful because it helps bridge the gap between ideal simulations and practical sensing systems. This enables the generation of more realistic synthetic datasets and improves the reliability of algorithms developed using transient rendering.
