message = "Beneath the moonlit sky, a shadowy figure emerged from the mist, moving with a purpose that betrayed years of training. In the heart of the city, secrets whispered through the alleys like an ancient melody, and the night air was thick with anticipation. The rendezvous was set, a meeting of minds veiled by the shroud of darkness"\
"Deep within the encrypted files lay the answers to a puzzle that had confounded even the most brilliant minds of our time.  Ciphers upon ciphers layered like the petals of an enigmatic flower guarded the truth, waiting for the intrepid soul who could unravel their intricacies The key to unlocking this cryptic tapestry remained hidden, passed down through generations of cryptographers. It was said that only one possessing unwavering determination and a keen eye for patterns could decipher the code. The fate of nations hung in the balance, and the clock ticked ominously as the world held its breath. As the first rays of dawn painted the sky, a breakthrough came. The faintest glimmer of understanding illuminated the edges of the intricate enigma. With each painstaking decryption, the fog of mystery lifted, revealing a trail of breadcrumbs leading to revelations beyond imagination. In the end, it was not just about decoding symbols on a page; it was about the journey of discovery, the unraveling of secrets, and the power that knowledge held. The shadows that once concealed truths gave way to the light of revelation, and a new chapter in the history of cryptography was written, forever altering the course of destiny."
import random

def encrypt(message):
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    map_alphabet = alphabet.copy()
    random.shuffle(map_alphabet)
    cypher = {}
    for i in range(len(alphabet)):
        cypher[alphabet[i]] = map_alphabet[i]
    encrypted_text = ""
    for ch in message:
        if ch.lower() in cypher:
            encrypted_ch = cypher[ch.lower()]
            encrypted_text += encrypted_ch
        else:
            encrypted_text += ch
    return encrypted_text
print(encrypt(message))