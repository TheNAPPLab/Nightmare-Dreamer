encrypted_text = "fwdwcta taw ymmdxzt iue, c iacpmre bzhnjw wywjhwp bjmy taw yzit, ymgzdh rzta c onjomiw tact fwtjcewp ewcji mb tjczdzdh. zd taw awcjt mb taw qzte, iwqjwti raziowjwp tajmnha taw cxxwei xzuw cd cdqzwdt ywxmpe, cdp taw dzhat czj rci tazqu rzta cdtzqzoctzmd. taw jwdpwlgmni rci iwt, c ywwtzdh mb yzdpi gwzxwp fe taw iajmnp mb pcjudwiipwwo rztazd taw wdqjeotwp bzxwi xce taw cdirwji tm c onllxw tact acp qmdbmndpwp wgwd taw ymit fjzxxzcdt yzdpi mb mnj tzyw.  qzoawji nomd qzoawji xcewjwp xzuw taw owtcxi mb cd wdzhyctzq bxmrwj hncjpwp taw tjnta, rcztzdh bmj taw zdtjwozp imnx ram qmnxp ndjcgwx tawzj zdtjzqcqzwi taw uwe tm ndxmquzdh tazi qjeotzq tcowitje jwyczdwp azppwd, ociiwp pmrd tajmnha hwdwjctzmdi mb qjeotmhjcoawji. zt rci iczp tact mdxe mdw omiiwiizdh ndrcgwjzdh pwtwjyzdctzmd cdp c uwwd wew bmj octtwjdi qmnxp pwqzoawj taw qmpw. taw bctw mb dctzmdi andh zd taw fcxcdqw, cdp taw qxmqu tzquwp myzdmnixe ci taw rmjxp awxp zti fjwcta. ci taw bzjit jcei mb pcrd oczdtwp taw iue, c fjwcutajmnha qcyw. taw bczdtwit hxzyywj mb ndpwjitcdpzdh zxxnyzdctwp taw wphwi mb taw zdtjzqctw wdzhyc. rzta wcqa oczditcuzdh pwqjeotzmd, taw bmh mb yeitwje xzbtwp, jwgwcxzdh c tjczx mb fjwcpqjnyfi xwcpzdh tm jwgwxctzmdi fwemdp zychzdctzmd. zd taw wdp, zt rci dmt knit cfmnt pwqmpzdh ieyfmxi md c ochw; zt rci cfmnt taw kmnjdwe mb pziqmgwje, taw ndjcgwxzdh mb iwqjwti, cdp taw omrwj tact udmrxwphw awxp. taw iacpmri tact mdqw qmdqwcxwp tjntai hcgw rce tm taw xzhat mb jwgwxctzmd, cdp c dwr qacotwj zd taw azitmje mb qjeotmhjcoae rci rjzttwd, bmjwgwj cxtwjzdh taw qmnjiw mb pwitzde."

def frequency_analysis(hidden_message):
    frequencies = {}
    num_ch = 0
    for ch in hidden_message:
        if ch.isalpha():
            num_ch += 1
            ch = ch.lower()
            if ch in frequencies:
                frequencies[ch] = frequencies[ch]+ 1
            else: 
                frequencies[ch] = 1

    sorted_frequencies = sorted(frequencies.items(), key=lambda item: item[1], reverse=True)
    return frequencies, num_ch, sorted_frequencies
frequencies, num_ch ,sorted_frequencies= frequency_analysis(encrypted_text)
print(sorted_frequencies)
