The final build. An on-call engineer at 3am needs answers from the payments runbook - and an assistant that invents procedures at 3am is dangerous. This assistant is graded as hard on what it refuses as on what it answers.


Build:

Index the runbook with the LangChain text splitter into a Chroma store at /root/runbook-assistant/chroma.
Write /root/runbook-assistant/assistant.py as a LangChain RAG chain (retriever, prompt template with both grounding rules, chat model), taking the question from sys.argv[1].
Run four questions and save each answer under /root/runbook-assistant/answers/:
"The payment queue is at 12000 messages, what do I do?" to queue.txt
"Certificate renewal failed, what is the fallback?" to cert.txt
"Who is on call this weekend?" to oncall.txt
"How do I refund a customer payment?" to refund.txt
The first two answers must carry the real numbers and steps from the runbook. The last two are not in the runbook at all - both must come back as the honest refusal.

Press Submit when the index, the assistant, and all four answers are in place.


