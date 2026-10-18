---
title: "{{ replace .Name "-" " " | title }}"
date: {{ .Date }}
draft: true
---

{{/*
Image shortcodes (images live next to index.md in the post bundle).
Captions accept markdown, e.g. *Fujifilm X-S10, 18-55mm*.

Single image wider than the text column; size = wide (~1100px, default) | full (edge to edge):
  {{< figure-wide src="photo.jpg" size="wide" caption="..." alt="..." >}}

2-3 images in one row at matching heights; one image per line, optional caption after "|".
Captions can go on each image, on the row, or both:
  {{< spread size="wide" caption="Caption for the whole row" >}}
  left.jpg | Caption for this image
  right.jpg
  {{< /spread >}}

Tweak widths/gap/caption display in assets/css/extended/custom.css (:root variables).
*/}}
