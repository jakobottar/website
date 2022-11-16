---
title: "Python Management with Pyenv"
date: 2022-08-17
draft: false
tags: ["Python"]
---

## Why use Pyenv?
Python virtual environments have been a useful way of managing python packages and package versions for quite a while. With vanilla Python, `virtualenv` is available, and for more complex cases Anaconda is a popular choice. Using these keeps your system installation of Python free of unnessecary clutter and packages as well as making it really easy to share dependencies with `pip freeze`. 

But what if some of the packages you wanted to install aren't avaliable for your system's installation of Python? Or what if your system is stuck on an old version of Python and you want to use the brand new shiny Python 3.13? That's where [pyenv](https://github.com/pyenv/pyenv) comes in. It's a series of shell 'shims' that seamlessly swap to different installations of Python. It's frequently updated which means new versions of Python get added regularly so you can always stay up to date. Even better, since it's simply shell commands you can install it without root access on your work machines!

## Installation
I suggest you skim through the ["How it works" section](https://github.com/pyenv/pyenv#how-it-works) of the GitHub repo to get a gist of pyenv's features and limitations. 

### Getting pyenv and pyenv-virtualenv
#### Linux
- Use your package manager of choice to install `pyenv` and `pyenv-virtualenv`
- Or check them out from GitHub ([pyenv](https://github.com/pyenv/pyenv) and [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv))

#### Mac
Install with Homebrew ([pyenv](https://github.com/pyenv/pyenv#homebrew-in-macos) and [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv#installing-with-homebrew-for-macos-users))

#### Windows
Sadly Windows is not offically supported. Your options are to either set up [WSL](https://docs.microsoft.com/en-us/windows/wsl/install) or to install the [Windows fork](https://github.com/pyenv-win/pyenv-win) (I haven't used this so have no experience with it). 

### Set up your Shell environment
The most complete examples can be found [here](https://github.com/pyenv/pyenv#set-up-your-shell-environment-for-pyenv), but here's instructions for the most common shells. Once you've added the following to your files, restart your shell with 
```bash
exec "$SHELL"
```

#### bash
Run this to add to your `~/.bashrc` file
```bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
```

then to add to your `~/.bash_profile` file, run
```bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
echo 'eval "$(pyenv init -)"' >> ~/.bash_profile
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bash_profile
```

#### Zsh
Run this to add to your `~/.zshrc` file
```zsh
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.zshrc
```

## Usage
You'll first want to install a distribution of Python, such as 3.10.4
```bash
pyenv install 3.10.4
```
You can list all available versions (theres a lot of them) with 
```bash
pyenv install --list
```

To set up a virtual environment named `my-env` on top of Python 3.10.4, run 
```bash 
pyenv virtualenv 3.10.4 my-env
```

To activate the virtual environment, use either 
- `pyenv shell <name>` -- select just for current shell session
- `pyenv local <name>` -- automatically select whenever you are in the current directory (and it's subdirectories)
- `pyenv global <name>` -- select globally for your user account

So, for example, in your project directory run 
```bash
pyenv local my-env
```
to set the environment in that directory (and it's sub-directories) to use your custom environment. Then go about using Python as normal, installing all your packages with `pip`! They're even cached on your machine so if you use the same package in two different environments there's no waiting to download them twice!
