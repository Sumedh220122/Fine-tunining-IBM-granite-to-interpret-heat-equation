# Fine-Tuning IBM Granite LLM for Solving the Heat Equation

This project aims to fine-tune the **IBM Granite Large Language Model (LLM)** to interpret and provide solutions to the **time-dependent heat equation**.

## Heat Equation Overview

The time-dependent heat equation is given as:

$$\frac{\partial T}{\partial t} - \alpha \left( \frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} \right) = f(x, y, t)$$


Where:
-  T: Temperature field
-  $$\alpha$$: Thermal conductivity (material property)
- f(x, y, t): Force function representing external influences

---

## Objectives

1. **Fine-tune IBM Granite LLM**: Train the model to interpret the heat equation and its boundary conditions for various scenarios.
2. **Solve Heat Equation**: Generate analytical or numerical solutions based on user-defined inputs for \( f(x, y, t) \), \( \alpha \), and initial/boundary conditions.

---

## Project Structure

