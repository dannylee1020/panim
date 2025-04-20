generate_inst_from_doc= """
Based on the following text extracted from Manim documentation, generate a diverse set of instruction-answer pairs suitable for supervised fine-tuning (SFT) an LLM whose goal is to generate Manim code based on user requests or descriptions.

**Core Objective:** Generate pairs that teach the target LLM *how to accomplish tasks* or *understand concepts* in Manim, drawing knowledge from the provided text. Frame instructions from the perspective of a user wanting to learn or use Manim, rather than simply asking what the documentation *says*.

**Requirements:**
1.  **Focus on Capabilities:** Ensure the generated pairs cover the *key capabilities, concepts, core functionalities, and primary code examples* presented in the text. Prioritize information that explains *how* to do something or *why* something works.
2.  **Generalized Instructions:** Instructions should be framed as general user queries, tasks, or requests for explanation, rather than questions *about the specific document text*.
    * **Good Example:** "How do I make an object fade out in Manim?" or "Generate Manim code for a fade-out animation."
    * **Less Ideal Example:** "What does this document say about the FadeOut animation?"
3.  **Task-Oriented Diversity:** Create pairs with varying levels of complexity and types of instructions, emphasizing practical application:
    * **Capability/How-To:** (e.g., "How do I plot a function?", "What's the way to change object color?")
    * **Code Generation/Examples:** (e.g., "Write Manim code to create a moving dot.", "Show an example of using the Text object.")
    * **Conceptual Understanding:** (e.g., "Explain the concept of an Mobject in Manim.", "What is the role of 'self.play()'?")
    * **Parameter/Configuration:** (e.g., "How can I set the radius of a Circle?", "What parameter controls animation speed?")
    * *(Optional)* Troubleshooting/Common Issues (if applicable): (e.g., "Why might text appear blurry?", "How to align objects vertically?")
4.  **Uniqueness:** Create pairs that are unique and cover distinct aspects of the information. Follow the DRY (Do not Repeat Yourself) principle across the generated pairs for the given text.
5.  **Clarity and Accuracy:** Instructions should be clear and unambiguous user queries/tasks. Answers should be accurate, concise, directly implement or explain the capability described in the text, and consist of explanatory text, Manim code, or both, as appropriate.
6.  **Format:** Output the result as a JSON list of objects. Each object must have exactly two keys: "instruction" (string) and "answer" (string, which may contain formatted code).

**Input Text:**
---
{input_text}
---

**Output (JSON List):**
"""

generate_inst_from_code = """
Based on the following Manim source code snippet, generate a diverse set of instruction-answer pairs suitable for supervised fine-tuning (SFT) an LLM whose goal is to generate Manim code based on user requests or descriptions of mathematical/visual concepts.

**Core Objective:** Analyze the provided Manim code to understand the **visual animation** it creates and/or the **mathematical concept** it illustrates. Then, generate pairs where the 'instruction' is a natural language description of that visual outcome or concept (phrased as a user request), and the 'answer' is the relevant code snippet itself. The goal is to teach the target LLM how to generate similar code when given such a description.

**Requirements:**
1.  **Infer Intent:** Analyze the code to determine its primary purpose. Focus on:
    * The overall visual transformation or animation sequence.
    * The specific mathematical concept being visualized (e.g., plotting a function, vector addition, geometric transformation).
    * Key Manim techniques or patterns demonstrated (e.g., using `Transform`, animating `ValueTracker`, creating complex layouts).
2.  **Generalized Task Instructions:** Formulate the 'instruction' as a clear, natural language description of the *task* the code performs or the *concept* it visualizes. Frame it as a request a user might make.
    * **Good Example (if code shows sin(x) plot):** "Generate Manim code to plot the sine function from x=0 to x=2*pi." or "Show how to animate the drawing of a sine wave."
    * **Less Ideal Example:** "What does this specific code snippet do?" or "Explain line 5 of this code."
3.  **Task-Oriented Diversity:** Create pairs reflecting different ways a user might request such an animation:
    * **Direct Code Generation Request:** (e.g., "Generate Manim code to show a circle transforming into a square.")
    * **Conceptual Visualization Request:** (e.g., "Create a Manim animation explaining the Pythagorean theorem visually.")
    * **Specific Technique Request:** (e.g., "Show how to use ValueTracker in Manim to animate a changing variable.")
    * **How-To Query:** (e.g., "How can I visualize matrix multiplication using Manim?")
4.  **Code as Answer:** The 'answer' field should contain the relevant Manim code snippet that fulfills the instruction.
    * If the input is a small, self-contained snippet, the answer might be the entire snippet.
    * If the input is a larger script chunk, the answer should be the minimal code section directly corresponding to the generated instruction. Ensure the code is runnable, potentially adding necessary context like imports or class/method definitions if implicitly needed and obvious, though primarily focus on the provided snippet.
5.  **Uniqueness:** If analyzing a larger script chunk by chunk, ensure pairs cover distinct visual/conceptual parts. Follow the DRY (Do not Repeat Yourself) principle for the capabilities demonstrated.
6.  **Clarity and Accuracy:** Instructions must clearly and accurately describe the visual/mathematical outcome achieved by the code. The code in the 'answer' must directly correspond to and fulfill the 'instruction'.
7.  **Format:** Output the result as a JSON list of objects. Each object must have exactly two keys: "instruction" (string) and "answer" (string containing Manim code).

**Input Code Snippet:**
---
```python
{source_code}
"""