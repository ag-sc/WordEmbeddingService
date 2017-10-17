# WordEmbeddingService
Build the project  -> mvn clean install

Train embeddings
	- place line separated sentences into e.g. sentenceCorpusEN, sentenceCorpusDE, sentenceCorpusES
	- java -jar target/WordEmbeddingService-0.0.1-jar-with-dependencies.jar -s train

Run APIs
	- java -jar target/WordEmbeddingService-0.0.1-jar-with-dependencies.jar -s api
	- to test "http://localhost:8080/similarity?word1=wann&word2=wie&lang=DE"
