supplement:
	rm -rf supplement
	mkdir supplement

	# make code directory
	mkdir supplement/code
	mkdir supplement/code/agents
	mkdir supplement/code/cfgs
	mkdir supplement/code/cfgs/agent
	cp README_supplement.md supplement/code/README.md
	# cp agents/* supplement/code/agents/
	cp agents/__init__.py supplement/code/agents/
	cp agents/td3.py supplement/code/agents/
	cp agents/iqrl.py supplement/code/agents/
	cp train.py supplement/code/train.py
	cp custom_types.py supplement/code/custom_types.py
	cp environment.yml supplement/code/environment.yml
	cp helper.py supplement/code/helper.py
	cp cfgs/agent/iqrl.yaml supplement/code/cfgs/agent
	cp cfgs/train.yaml supplement/code/cfgs
	cp post-install-amd.txt supplement/code
	cp -r cfgs/env supplement/code/cfgs
	cp -r utils supplement/code/

	rm -f supplement.zip
#	zip supplement.zip 'supplement/supplement.pdf'
	zip -r supplement.zip supplement/code -i '*.py'
	zip -r supplement.zip supplement/code -i '*.txt'
	zip -r supplement.zip supplement/code -i '*.md'
	# zip -r supplement.zip supplement/code -i '*.ipynb'
