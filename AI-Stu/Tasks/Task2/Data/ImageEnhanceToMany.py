from PIL import ImageEnhance


def ImageEnhanceToMany(image, fromRate, toRage):
    enhancer = ImageEnhance.Brightness(image)
    tagRate = fromRate
    outPut = []
    while tagRate <= toRage:
        if (tagRate == 1):
            outPut.append(image)
        else:
            brightened_image = enhancer.enhance(tagRate)
            outPut.append(brightened_image)
        tagRate += 0.1
    return outPut
