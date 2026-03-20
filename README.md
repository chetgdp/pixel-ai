# pixel.ai
Make a single pixel generator AI model, anything in -> one pixel out

A pixel is made up of 3 or 4 u8 values: 
- RGB/RGBA (Red, Green, Blue, Alpha)
- HSV/HSL (Hue, Saturation, Value/Lightness)
- CMYK (Cyan, Magenta, Yellow, blacK)

For our purposes we can do RGBA, it is quite simple to perform conversions anyway. So we have our output, one pixel, 4 bytes, thats a u32

Anything in means we should accept an array of bytes as input: [u8]

text? ascii, utf8, utf16, easy. images? video? documents? simple, just read the bytes

How big of an input should we accept? a few mb? No, larger, say 4gb? or essentially whatever the static maximum is. 

This can be a very simple neural network. You have 4 output nodes and many many more input nodes. 





