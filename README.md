# AI-Projec
Processing the data:
1. We get MIDI files, store them in a folder and then iterate over them.
2. Each file we convert to a piano roll using the library pretty_midi. A piano roll contains 1/100 sec time
frames, where each frame is a vector of size 128 representing the different pitches (extra entries for
padding to reach power of 2). Each entry contains the key velocity at that time frame.
3. We saw lots of identical consecutive time frame, so to compress the data and help the model
understand the data better, we decided to replace cuncurent duplicates with one time frame where the
last entry contains the number of duplicates.
4. Next we faced a problem of how to store the pieces. We noticed that most notes are rarely used,
so the entries in the piano roll are mostly zeros. We used that information to compress the data,
and then we stored the result. We got that the processed data ended up taking less then half the space
of the original midi file. For example, a 9:43 minutes piano piece of size 68KB was processed into a
piano roll of size 28KB.
5. To store many piano rolls we simply concatenate the piano rolls. (OR) We store different files
corresponding to the piece. We also added an end of piano roll time frame, so that the model would
also know to predict an end of song.


TODO:
Cast the data to float before feeding to the nn.
