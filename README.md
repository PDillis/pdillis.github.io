# Welcome to my blog

I will be using the [Jekyll][1] theme for [@jasonlong][2]'s [Cayman theme][4] on [GitHub Pages][3], with some slight modifications (mostly aesthetic). I will leave unchanged the instructions on how to use it.

## How to use it?

Download the theme @ https://github.com/pietromenna/jekyll-cayman-theme/archive/master.zip

Unzip it and use it as a regular jekyll folder.

```
$ unzip jekyll-cayman-theme-master.zip
```

Get inside the newly extracted folder
```
$ cd jekyll-cayman-theme-master
```

Get the required gems
```
$ bundle install
```

Use it!

```
$ jekyll serve
```

However, due to recent updates and the requirements to [using Bundler 1.12](https://github.com/PDillis/pdillis.github.io/blob/ac44808e7d7bef62281d9646c573e96eebce20e3/jekyll-cayman-theme.gemspec#L16) in the selected Cayman Theme, we must then install this version of Bundler via:

```
gem install bundler -v 1.12
```

Then we can install the required gems:

```
bundle _1.12_ install
```

It's normal to have errors in the required/installed/activated packages, so instead of the `jekyll serve` instruction as before, [the correct way](https://stackoverflow.com/a/6393129) would be to use:

```
bundle exec jekyll serve
```

Now, we can just go to `http://127.0.0.1:4000` and see our website *live*, instead of doing unnecessary commits, or waiting for GitHub pages to do their thing.

For more details read about [Jekyll][1] on its web page.

## Setup

Some important configuration can be done in the file `_config.yml`. To use the Cayman theme and use Markdown, I have added the following:

```
markdown: kramdown
theme:    jekyll-theme-cayman
```

# License

This work is licensed under a [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) license.

[1]: https://jekyllrb.com/
[2]: https://github.com/jasonlong
[3]: https://pages.github.com/
[4]: https://github.com/jasonlong/cayman-theme
