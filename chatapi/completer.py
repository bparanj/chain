from completion import get_completion

result = get_completion("Bugs Bunny is ")

print("\n" + "="*50 + "\n")
print(type(result))
print("\n" + "="*50 + "\n")

from pprint import pprint
pprint(vars(result), indent=2, width=80)

print("\n" + "="*50 + "\n")

answer = result.choices[0].message.content

print(answer)
