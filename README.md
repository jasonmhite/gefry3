Gefry
=====
This is a package to approximate the response of radiation detectors
to a gamma source in a heterogeneous environment. Read any of the papers
I've written recently for details, because I doubt you got here by
accident.

Input deck
----------
Gefry reads an input deck that describes the model geometry. This input
deck is written in [YAML](https://en.wikipedia.org/wiki/YAML), a simple
markup format. It's pretty self explanatory and you shouldn't need to
change stuff much.

Geometry
--------
All geometry is represented as polygons (curves have to be approximated).
These polygons are described by lists of vertices. Each polygon also
has an associated material, with the materials described in their own
section. Each material is characterized by a number density and a 
*micro*scopic cross section. If you want to assign *macro*scopic
cross sections then set number density to 1.0. Materials can be reused.
There is also an "interstitial material", which is the material of everything
that is not one of the defined regions (i.e., the air). 

**NOTE** the order of the regions is important! If you want to run a 
problem with varying cross sections, the order of the cross sections you
provide must match the order of polygons in the input file!

How to use and caveats
----------------------
This way to use this is read an input deck, which returns an object
that can be called to evaluate the counts seen by the detector. The arguments
you provide vary depending on what kind of problem you are running,
currently either `Simple_Problem` (fixed cross sections, you just provide
the source location and intensity) or `Perturbable_XS_Problem` (variable
cross sections, you provide source location and intensity, plus a list
of cross sections).

The biggest thing to know is that this version of the code is *purely deterministic*.
It only evaluates the ray tracing model for the detector network and does
not include any statistical effects or background. You add those on your own,
however you desire.

A note on Python versions
-------------------------
I develop and use this code on Python 3. NumPy support for Python 2.7 is being dropped at the
end of 2018, so Python 2.7 is dead and it's time to move on. The code should be compatible,
but I don't really test it on Python 2 so you might encounter bugs. If you do let me know and
I will try to fix them unless it's really hopeless.
