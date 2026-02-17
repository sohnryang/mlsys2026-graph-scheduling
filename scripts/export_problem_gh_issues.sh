#!/bin/bash
gh issue view $1 -R yarongmu-google/MLSys --json number,title,url,author,createdAt,body,comments | jq -r '
  def md_escape: gsub("\r";"");
  "---\nsource: github\nrepo: owner/repo\nissue_number: \(.number)\nissue_title: \"\(.title|gsub("\"";"\\\""))\"\nissue_url: \(.url)\nexported_at: \(now|todateiso8601)\n---\n\n" +
  "# Issue #\(.number): \(.title)\n\n" +
  "## Original post\n" +
  "- author: \(.author.login)\n" +
  "- created_at: \(.createdAt)\n" +
  "- url: \(.url)\n\n" +
  "<comment>\n\(.body|md_escape)\n</comment>\n\n---\n\n" +
  (
    .comments
    | to_entries
    | map(
        "## Comment \(.key+1)\n" +
        "- author: \(.value.author.login)\n" +
        "- created_at: \(.value.createdAt)\n" +
        "- url: \(.value.url)\n\n" +
        "<comment>\n\(.value.body|md_escape)\n</comment>\n\n---\n"
      )
    | join("\n")
  )
'
