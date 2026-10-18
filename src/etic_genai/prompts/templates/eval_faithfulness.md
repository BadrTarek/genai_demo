You are a strict evaluator checking an AI answer for hallucinations.

Score how well every factual claim in the ANSWER is supported by the CONTEXT passages.
- 1.0: every claim is directly supported by the context.
- 0.5: the main claim is supported but some details are not.
- 0.0: the main claim is unsupported or contradicts the context.
Ignore citation markers such as [1]. Do not use outside knowledge. An answer that says the
information could not be found is fully faithful.

QUESTION:
{question}

CONTEXT:
{contexts}

ANSWER:
{answer}
