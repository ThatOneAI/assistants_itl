import asyncio

import hfa_itls
import tool_itls

# All the actual work is done by hfa_itls.py and tool_itls. We just need to wait and
# let them do their thing.


async def main():
    while True:
        await asyncio.sleep(999)


if __name__ == "__main__":
    asyncio.run(main())


"""
1. start the ui, explain that it's a ui
2. start the script, explain
3. add the config files, explain
4. get the contents of a story, show the generated code, explain that this is what hf agents does, you only modified it to make it dynamically configurable, and added a wrapper to talk to it via websockets
5. show the code for making it configurable (maybe just the prompt)
6. show getting dialogue lines
7. modify extract-dialogue-v1.yaml so it uses 'edit' to output as a chat log
8. rerun the command to show the change
"""
