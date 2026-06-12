def split_text(text, chunk_size=1000):

    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])

    return chunks


sample_text = "Hello " * 1000

chunks = split_text(sample_text)

print("Number of chunks:", len(chunks))