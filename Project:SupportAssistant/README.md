A support inbox has waiting tickets, and a knowledge base has the answers. Build the assistant that connects them: for each ticket, find the right article and write a reply grounded in it.

Build:

Write /root/support-assistant.py. It embeds the articles once, then for every ticket in /root/tickets: embeds the ticket, retrieves the best-matching article, and asks a model to write a short support reply using only that article for facts.
Save each reply to /root/replies/ with the same file name as its ticket (ticket-1.txt and so on).
Careful with ticket 2: it asks for a copy of stored data. The deletion article is a close neighbor on the meaning map - your retrieval must pick the export article, and the reply must describe the export.

Press Submit when the script and all three replies are in place.

