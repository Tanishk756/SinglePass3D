# Scientific and operational limitations

One flight path cannot measure unseen surfaces. Occluded facades, undersides, and interiors remain unknown. Limited parallax or nearly collinear camera centers can prevent a stable full 3D alignment. Low overlap, blur, weak or repeated texture, shadows, reflective surfaces, and moving objects can impair COLMAP.

GPS may be noisy or delayed. The current pipeline requires an explicit video-to-UTC start time and does not solve clock offset, antenna lever arm, or altitude datum conversion. Single-view depth would have scale ambiguity and must be marked inferred. Sparse PLY points are observed geometry, but their georeferencing inherits GPS and alignment uncertainty. Dense geometry, semantic masking, meshing, GLB, and viewer are not yet implemented.
