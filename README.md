# SLM-for-Academic-Calendar-Interpretation
FOR NLP Project

Finding a specific deadline or start date in an academic calendar often means digging through a long, confusing document. This process is slow and it's easy to get the dates wrong. This project aims to solve that problem by building a Small Language Model (SLM) from scratch, designed specifically to understand university schedules. It will be able to answer simple questions like, "When is the last day to drop a class?" or "When does the fall semester start?".

Unlike projects that use huge, general-purpose AI, our focus is on creating a lightweight and specialized model. This approach makes the tool cheaper to run, protects student data, and makes it easy to adapt for any university.

Our plan starts with gathering and cleaning up calendar data from various universities. We'll then create a custom vocabulary and word system (embeddings) that knows the specific language of academia. Using that data, we'll train a compact model with proven methods like RNNs and BiLSTMs. The final model will be tuned to run efficiently on basic hardware while still being excellent at understanding questions and extracting the correct dates.

The goal is to produce a small, fast, and reliable model that helps eliminate student confusion and makes administrative work easier. This project will show that specialized SLMs are a great alternative to giant language models, offering a path to AI solutions that are private, resource-friendly, and provide great hands-on learning.
