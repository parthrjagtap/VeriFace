$(document).on("change", "#id_upload_video_file", function (evt) {
    var $source = $('#video_source');
    $source[0].src = URL.createObjectURL(this.files[0]);
    $source.parent()[0].load();
    $('#videos').css("display", "block");
    $('#id_upload_video_file').css("display", "none");
});
$('form').on('submit', function (e) {
    $('#videoUpload').prop("disabled", true);
    $('#videoUpload').html('Uploading Video&nbsp;<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span><span class="sr-only">Loading...</span>');
});