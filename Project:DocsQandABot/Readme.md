Build a bot that answers questions about a team's documentation folder. This is the full course in one tool: read files, chunk, embed, store, retrieve, ground, refuse.

Seed the documents first:

bash

curl -fsSL https://raw.githubusercontent.com/Esc-Bash/project-init-scripts/main/rag/project-1/init.sh | bash
This creates /root/docs/ with four markdown files: deploys, backups, access, and incidents.

Build:

Index the folder: read every file in /root/docs, chunk the text, embed the chunks, and store them. Use Chroma at /root/docsbot/chroma, or your own index under /root/docsbot/index - your choice.
Write /root/docsbot/ask_docs.py: takes a question as a command-line argument, retrieves the closest chunks, builds the grounded prompt with both rules, prints the answer.
Run three questions and save the outputs:
"Which days does the web app deploy?" to /root/docsbot/answer-deploy.txt
"How long are database backups kept?" to /root/docsbot/answer-backup.txt
"How do I connect to the VPN?" to /root/docsbot/answer-vpn.txt - no document covers the VPN, so this must be the honest refusal.
Press Submit when the index, the script, and all three answer files are in place.
