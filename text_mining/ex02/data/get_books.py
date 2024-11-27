import requests
import tqdm

books_de = [6975, 6996, 7276, 9875, 7185, 6990, 6924, 2229, 2403, 2228]
books_fr = [13868, 18143, 15032, 15847, 18179, 15303, 26759, 54202, 10604, 17509]

for idx, (de, fr) in tqdm.tqdm(enumerate(zip(books_de, books_fr))):
    resp_de = requests.get(f"https://gutenberg.org/cache/epub/{de}/pg{de}.txt")
    resp_fr = requests.get(f"https://gutenberg.org/cache/epub/{fr}/pg{fr}.txt")
    assert (
        resp_de.ok and resp_fr.ok
    ), f"{resp_de.status_code} for {de}, {resp_fr.status_code} for {fr}"

    with open(f"de/{idx+1:0>2}_{de}.txt", "w") as output:
        output.write(resp_de.text)

    with open(f"fr/{idx+1:0>2}_{fr}.txt", "w") as output:
        output.write(resp_fr.text)
