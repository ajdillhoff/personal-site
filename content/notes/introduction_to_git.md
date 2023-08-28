+++
title = "Introduction to Git"
authors = ["Alex Dillhoff"]
date = 2023-08-26T00:00:00-05:00
tags = ["git", "programming"]
draft = false
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [What is Version Control?](#what-is-version-control)
- [What is Git?](#what-is-git)
- [What is a Repository?](#what-is-a-repository)
- [Configuring Git](#configuring-git)
- [Creating a Repository](#creating-a-repository)
- [Staging Files](#staging-files)
- [Committing Changes](#committing-changes)
- [Ignoring Files](#ignoring-files)
- [Branching](#branching)
- [Merging](#merging)
- [Remotes](#remotes)
- [Cloning an Existing Repository](#cloning-an-existing-repository)
- [Summary](#summary)

</div>
<!--endtoc-->



## What is Version Control? {#what-is-version-control}

Version control is a system that records changes to a file or set of files over time so that you can recall specific versions later. This is useful not just for team projects, for for individual projects as well.

With version control, you can:

-   revert files back to a previous state
-   revert the entire project back to a previous state
-   compare changes over time
-   see who last modified something that might be causing a problem
-   who introduced an issue and when
-   and much more

When tracking changes to a project over time, the simplest approach is one that you might recognize if you've ever worked on an essay for class. Imagine you've just finished the first draft of an assignment. You decide to save this document as `essay_first_draft.docx`. After working on it a bit more, you choose to save the updated copy to a new file so that you can compare the initial and final draft. This one is then named `essay_first_draft_COMPLETE.docx`. You end up reading some new information and realize you missed a key requirement of the assignment. After adding in the new information you save it as `essay_first_draft_COMPLETE_v2.docx`. After many such iterations you end up with a collection of ill-named files.

Maybe you've never done this yourself, but this example actually depicts a version control system. Luckily for us, there have been many improvements to this naive method. A more ideal choice, especially in a team environment, would be a Centralized VCS. Project files would be hosted on a server that keeps track of the different changes. Team members can download the latest versions, modify them, and update the server once they are done.

{{< figure src="/ox-hugo/2023-08-27_18-53-58_screenshot.png" caption="<span class=\"figure-number\">Figure 1: </span>Centralized VCS ([source](<https://git-scm.com/book/en/v2/Getting-Started-About-Version-Control>))" >}}

This is a welcome improvement over the naive version, but it still has its downsides. What if the server or connection goes down? There are many scenarios that would lead to a catastrophic loss of data. For important projects, you would not want to keep all of your eggs in once basket. A more ideal solution would be a Distributed VCS, this is what Git is.

{{< figure src="/ox-hugo/2023-08-27_18-58-25_screenshot.png" caption="<span class=\"figure-number\">Figure 2: </span>Distributed VCS ([source](<https://git-scm.com/book/en/v2/Getting-Started-About-Version-Control>))" >}}

In a DVCS, every user has a complete copy of the project. If the server goes down, or a connection is lost, you can still work on the project. Since every user has a complete copy of the project, there is no single point of failure. Another huge advantage is speed. The operations are performed locally. It is only when you want to share your changes that you need to connect to the server. This means that you can commit changes, create branches, and perform other operations without an internet connection.


## What is Git? {#what-is-git}

There are two primary ways of thinking about versioning in general: snapshots and differences. The first starts with your original files and records each change as a delta between the latest version and the previous.

{{< figure src="/ox-hugo/2023-08-27_19-03-32_screenshot.png" caption="<span class=\"figure-number\">Figure 3: </span>Difference-based Version Control ([source](<https://git-scm.com/book/en/v2/Getting-Started-What-is-Git%3F>))" >}}

The second starts with your original files and records each change as a snapshot of the entire project. Files that have not changed will not be duplicated. Instead, Git will create a reference to the previous version of the file. This is the approach that Git uses, and it comes with a great benefit that we will see when we get to branching.

{{< figure src="/ox-hugo/2023-08-27_19-05-52_screenshot.png" caption="<span class=\"figure-number\">Figure 4: </span>Snapshot-based Version Control ([source](<https://git-scm.com/book/en/v2/Getting-Started-What-is-Git%3F>))" >}}


## What is a Repository? {#what-is-a-repository}

A repository is a collection of files and folders that are tracked by Git. It is the project folder that you will be working in. You can create a repository from scratch, or you can clone an existing repository. We will cover examples of both during class. Cloning is the process of copying an existing repository to your local machine. We will start with the first approach: creating a repository from scratch.

Before starting, it is important to at least know the three major states of Git. Files can be `modified`, `staged`, or `committed`.

-   A `modified` file has been changed locally, but has not been committed to the repository.
-   A `staged` file is a modified file that has been marked to be included in the next commit.
-   A `committed` file is a staged file that has been saved to the repository.

{{< figure src="/ox-hugo/2023-08-27_19-16-09_screenshot.png" caption="<span class=\"figure-number\">Figure 5: </span>The main sections of Git. ([source](<https://git-scm.com/book/en/v2/Getting-Started-What-is-Git%3F>))" >}}

The figure above depicts the three major sections of working with a Git repository. Each repository has a `.git` directory that contains all of the information about the project. The `working directory` is the root directory where the latest versions of the files exist. Once modifications are made, these changes are sent to the `staging area`. This is where you can choose which changes to include in the next commit. Once you are happy with the changes, you can commit them to the repository. This will save the changes to the `.git` directory.


## Configuring Git {#configuring-git}

Once you have installed Git, there are a few important configuration options to get started. If you have already been using Git, you can skip this section. If you are using Git for the first time, you will need to set your name and email address. This information will be used to identify you as the author of the commits that you make.

```bash
git config --global user.name "Naomi Nagata"
git config --global user.email "naomi@rocinante.exp"
```

If you have already used a service like GitHub, note that this name and email does not need to match the one you used to log into that service.

You can view your current configuration at any time by running the following command:

```bash
git config --list --show-origin
```


## Creating a Repository {#creating-a-repository}

For this example, our first project will be a Python program that resizes images to a specified width. This is to ensure that the aspect ratio is maintained.

Now that we have Git installed and configured, we can create our first repository. First, create a new directory for the project. I will use `pyresize` in this document. Then, navigate to that directory and run the following command:

```bash
mkdir pyresize && cd pyresize
git init
```

You may see the following warning when creating a new repository:

```bash
hint: Using 'master' as the name for the initial branch. This default branch name
hint: is subject to change. To configure the initial branch name to use in all
hint: of your new repositories, which will suppress this warning, call:
hint:
hint: 	git config --global init.defaultBranch <name>
hint:
hint: Names commonly chosen instead of 'master' are 'main', 'trunk' and
hint: 'development'. The just-created branch can be renamed via this command:
hint:
hint: 	git branch -m <name>
Initialized empty Git repository in /home/alex/dev/pyresize/.git/
```

Let's first set the default branch name to `main`.

```bash
git config --global init.defaultBranch main
```

Next, we will change the name of the current branch to `main`. We could also delete the `.git` directory and start over, but this is a good opportunity to learn how to rename a branch.

```bash
git branch -m main
```

You can view the status of your repository at any time by using the `git status` command. This will show you the current branch, the files that have been modified, and the files that have been staged. Our newly created repository looks like this:

```bash
$ git status
On branch main

No commits yet

nothing to commit (create/copy files and use "git add" to track)
```


## Staging Files {#staging-files}

Let's create our first file and add it to the repository. We will create a file called `pyresize.py` that contains the following code:

```python
import sys
from PIL import Image

def resize_image(image_path, width):
    image = Image.open(image_path)
    wpercent = (width / float(image.size[0]))
    hsize = int((float(image.size[1]) * float(wpercent)))
    image = image.resize((width, hsize), Image.LANCZOS)
    image.save(image_path)

if __name__ == "__main__":
    resize_image(sys.argv[1], int(sys.argv[2]))
```

At this point, we have a local change that our repository is not aware of. We can see this by running the `git status` command again.

```bash
$ git status
On branch main

No commits yet

Untracked files:
  (use "git add <file>..." to include in what will be committed)
    pyresize.py

nothing added to commit but untracked files present (use "git add" to track)
```

Let's add our file with `git add pyresize.py` and check the status again.

```bash
$ git add pyresize.py
$ git status
On branch main

No commits yet

Changes to be committed:
  (use "git rm --cached <file>..." to unstage)
    new file:   pyresize.py
```


## Committing Changes {#committing-changes}

Finally, we will `commit` this change via `git commit`. There are a few things to note about this command. If you haven't configured your default editor, it might be set to something like `nano` or `vim` by default. If you are not familiar with these editors, you can set your default editor to something else by running the following command:

```bash
git config --global core.editor "code --wait"
```

This will set your default editor to Visual Studio Code. Obviously, this should be installed on your system if you are using it. If you are using a different editor, you can replace `code` with the command that you would use to open a file in that editor. For example, if you are using `vim`, you would use `vim`. The `--wait` flag above will wait for the editor to close before continuing. This is important for Git to know when you are done writing your commit message. **Note that not every application supports this flag.**

Once you have set your default editor, you can run `git commit` to open the editor and write your commit message. The first line should be a short description of the change. The following lines should be a more detailed description of the change. You can see an example below:

```git
Added our first file.
# Please enter the commit message for your changes. Lines starting
# with '#' will be ignored, and an empty message aborts the commit.
#
# On branch main
#
# Initial commit
#
# Changes to be committed:
#	new file:   pyresize.py
#
```

Once we save this message, the commit will be complete. You can view the commit history by running `git log`. This will show you the commit hash, the author, the date, and the commit message. You can also commit changes and add a message in one command.

```bash
git commit -m "Added our first file."
```

We do not yet have a remote repository to `push` to, so we will save that for later. For now, we will continue to work locally. Let's add an image to our repository so that we can test it. I am going to use the [UTA Logo](![](https://resources.uta.edu/mme/identity/_images/new-logos/new-initials-logo.jpg)) for this example. You can download this and variations from the [UTA Branding Resources](<https://resources.uta.edu/mme/identity/brand/index.php>) page.

{{< figure src="/ox-hugo/2023-08-27_20-04-35_screenshot.png" caption="<span class=\"figure-number\">Figure 6: </span>UTA Logo" >}}

Create a new `imgs` folder and add your image(s) to it. We can then add the directory along with all of its contents using `git add imgs`. You can see the status of your repository by running `git status` again. Let's go ahead and `commit` these changes.

```bash
git commit -am "Added the UTA logo."
```


### Making a Change {#making-a-change}

For this project, we don't really need to have a bunch of test images. It is sufficient to have one or two. The name of our image folder should probably change to reflect its purpose. Let's start by renaming `imgs` to `test_imgs`. We can do this with the `mv` command in bash. Our repository will now look like this:

```bash
$ git status
On branch main
Changes not staged for commit:
  (use "git add/rm <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
	deleted:    imgs/uta.png

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	test_imgs/

no changes added to commit (use "git add" and/or "git commit -a")
```

This might be a tad unexpected. We renamed the folder, but Git is telling us that we deleted a file. This is because Git is tracking the file `imgs/uta.png`. When we renamed the folder, Git no longer knew where to find the file. We can fix this by running `git rm imgs/uta.png`. This will remove the file from the repository. We can then add the new folder with `git add test_imgs`. However, if we simply use `git add test_imgs`, Git will not know that we renamed the folder. We can fix this by using the `-A` flag. This will tell Git to add all changes, including renames. Our repository will now look like this:

```bash
$ git status
On branch main
Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
    renamed:    imgs/uta.png -> test_imgs/uta.png
```

Go ahead and `commit` these changes.


## Ignoring Files {#ignoring-files}

There are certain files and directories that will end up in our project folder that we do not want to track. For example, we may want to resize images and save them in a local `output` directory. However, we do not want to track any of these images. We can have Git remember what we want or do _not_ want using an ignore file. This file will contain a list of files and directories that we want to ignore. Let's create a `.gitignore` file and add the following line:

```bash
output/
```

Make sure you add and commit the `.gitignore` file.


## Branching {#branching}

Let's go ahead and test our program by resizing one of our images. I'm going to to resize `uta.png` to have a width of 500 pixels using the following command.

```bash
python pyresize.py test_imgs/uta.png 500
```

Our program doesn't support creating a new file when resizing. Instead, it resizes the file and overwrites the original. We should add this feature and modify it without messing up our current code base. This is where branching comes in. We can create a new branch that will contain our new feature. We can then test it and merge it back into the main branch once we are happy with it.

Every `commit` that we make is a snapshot of the entire project up to that point. There is a unique identifier attached to each commit. If we want to work on a specific bug or new feature without affecting the current code base, we can create a branch to track those changes independently of the other branches. The main benefits are that we can potentially break the code base without affecting the production-ready code. We can also work on multiple features at the same time without affecting each other.

Let's create a new branch called `output_write`. We can do this with the `git branch` command.

```bash
git branch output_write
```

{{< figure src="/ox-hugo/2023-08-27_20-38-44_screenshot.png" caption="<span class=\"figure-number\">Figure 7: </span>The result of creating a new branch named `testing`. ([source](<https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell>))" >}}

This only creates a new branch, but we are still on the `main` branch. We can see this by running `git branch`. The current branch will be highlighted with an asterisk. The current branch is pointed to by the `HEAD` pointer. We can switch to the new branch using `git checkout`.

```bash
git checkout output_write
```

{{< figure src="/ox-hugo/2023-08-27_20-40-25_screenshot.png" caption="<span class=\"figure-number\">Figure 8: </span>The `HEAD` pointer after switching to a new branch. ([source](<https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell>))" >}}


### Modifying the Code {#modifying-the-code}

Now that we are on the `output_write` branch, we can modify the code without affecting the `main` branch. Let's modify our original function to take in an additional argument: the output path. We can then use this path to save the resized image to a new file. Our new code will look like this:

```python
import sys
from PIL import Image

def resize_image(image_path, width, output_path):
    image = Image.open(image_path)
    wpercent = (width / float(image.size[0]))
    hsize = int((float(image.size[1]) * float(wpercent)))
    image = image.resize((width, hsize), Image.LANCZOS)
    image.save(output_path)

if __name__ == "__main__":
    resize_image(sys.argv[1], int(sys.argv[2]), sys.argv[3])
```

Go ahead and commit these changes. Since we have already moved the `HEAD` pointer to the new branch, this change will not affect our `main` branch. The figure below is analagous to this scenario.

{{< figure src="/ox-hugo/2023-08-27_20-45-17_screenshot.png" caption="<span class=\"figure-number\">Figure 9: </span>The result of committing changes to a new branch. ([source](<https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell>))" >}}

Let's test our program once more by resizing an image and saving it to the `output` directory. We can do this with the following command:

```bash
python pyresize.py test_imgs/uta.png 500 output/uta.png
```

Notice that when you run `git status` after resizing the image and saving to the `output` directory, it does not report any changes. Our ignore file is working as intended!


## Merging {#merging}

Now that we have completed our new feature and tested it, we should merge these changes back to the `main` branch. We can do this with the `git merge` command. First, we need to switch back to the `main` branch.

```bash
git checkout main
```

We can then merge the `output_write` branch into the `main` branch.

```bash
git merge output_write
```

This will merge the changes from the `output_write` branch into the `main` branch. If there are any conflicts, Git will let you know and you can resolve them manually. Once the merge is complete, you can delete the `output_write` branch.

```bash
git branch -d output_write
```

{{< figure src="/ox-hugo/2023-08-27_20-51-26_screenshot.png" caption="<span class=\"figure-number\">Figure 10: </span>The result of merging a branch into the `master` branch. ([source](<https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell>))" >}}

The figure above shows the history of a repository in which a branch named `iss53` was created, modified with new commits, and eventually merged back into the `master` branch.


## Remotes {#remotes}

We have now covered the basics of using Git locally. Eventually, we will want our changes to be backed up on a remote server. This will allow us to collaborate with others and work on our projects from multiple machines. There are many services that provide this functionality. We will use GitHub for this example, but the process is similar for other services.


### Creating a Repository {#creating-a-repository}

First, we need to create a new repository on GitHub. You can do this by clicking the `New` button on the [GitHub homepage](<https://github.com>). I am only going to add a short description of this program. Go ahead and click `Create repository`.

GitHub supports both SSH and HTTPS. I already have an SSH key set up. If you haven't configured one yet, check out [Adding a new SSH key to your GitHub account](<https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account>) for instructions on how to do so. You can also use HTTPS, this requires a personal access token. More information can be found at [Creating a personal access token](<https://docs.github.com/en/github/authenticating-to-github/creating-a-personal-access-token>).

Once created, we will have the option to use either our HTTPS or SSH URL. Mine is `git@github.com:ajdillhoff/pyresize.git`. We can add this as a remote repository using the `git remote add` command.

```bash
git remote add origin git@github.com:ajdillhoff/pyresize.git
```


### Pushing to a Remote {#pushing-to-a-remote}

Now that we have a remote repository, we can push our changes to it. We can do this with the `git push` command. However, we need to specify the remote repository and the branch that we want to push. We can do this with the following command:

```bash
git push -u origin main
```

That's it! Our changes are now backed up on GitHub. We can view our repository by navigating to the URL in our browser. We can also view the commit history by clicking the `Commits` link.


## Cloning an Existing Repository {#cloning-an-existing-repository}

If our repository already exists on GitHub, we can clone it to our local machine. This will create a new directory with the same name as the repository. We can do this with the `git clone` command. Let's clone the `pyresize` repository that we just created.

```bash
git clone git@github.com:ajdillhoff/pyresize.git
```

You can clone using either the HTTPS or SSH URLs. Make sure you have the appropriate key or access token to do so.


### Pulling from a Remote {#pulling-from-a-remote}

Now that we have cloned the repository, we can make changes and push them to the remote. However, if someone else makes changes to the remote repository, we will need to pull those changes to our local repository. We can do this with the `git pull` command.

```bash
git pull origin main
```

Git will always require that we are up-to-date with the remote before we can push our changes. If someone else has made changes to the remote, we will need to pull those changes before we can push our own. This is to prevent conflicts.


## Summary {#summary}

We have covered the basics of using Git. We have created a repository, staged and committed changes, created branches, merged branches, and pushed our changes to a remote repository. There are many more features that we have not covered, but this should be enough to get you started. If you are interested in learning more, check out the [Git Book](<https://git-scm.com/book/en/v2>).

**Command Reference**

| Command          | Description                                                        |
|------------------|--------------------------------------------------------------------|
| `git init`       | Create a new repository                                            |
| `git config`     | Configure Git                                                      |
| `git status`     | View the status of your repository                                 |
| `git add`        | Add files to the staging area                                      |
| `git commit`     | Commit changes to the repository                                   |
| `git branch`     | Create, list, or delete branches                                   |
| `git checkout`   | Switch branches or restore working tree files                      |
| `git merge`      | Join two or more development histories together                    |
| `git remote`     | Manage set of tracked repositories                                 |
| `git push`       | Update remote refs along with associated objects                   |
| `git clone`      | Clone a repository into a new directory                            |
| `git pull`       | Fetch from and integrate with another repository or a local branch |
| `git log`        | Show commit logs                                                   |
| `git rm`         | Remove files from the working tree and from the index              |
| `git mv`         | Move or rename a file, a directory, or a symlink                   |
| `git branch -d`  | Delete a branch                                                    |
| `git remote add` | Add a remote repository                                            |
